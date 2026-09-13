"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — Trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và tiếp nhận yêu cầu hỗ trợ khách hàng cho hệ sinh thái Vingroup (VinFast, Vinpearl, VinWonders, Landmark 81,...).
- Phong cách giao tiếp: Chuyên nghiệp, lịch sự, thân thiện, trung thực và chính xác tuyệt đối.

## 2. AVAILABLE TOOLS
- `search_product_catalog(category, max_price)`: Tra cứu danh mục sản phẩm và dịch vụ của Vingroup theo phân loại ('xe_dien' hoặc 'du_lich') và mức giá tối đa (VNĐ).
- `submit_support_ticket(customer_name, issue_description, priority)`: Ghi nhận thông tin phản ánh, khiếu nại hoặc sự cố kỹ thuật của khách hàng vào hệ thống ticket hỗ trợ.

## 3. CORE RULES
- KHÔNG BAO GIỜ tự bịa đặt dữ liệu sản phẩm, giá bán, hoặc thông số kỹ thuật (Zero Hallucination).
- BẮT BUỘC phải gọi công cụ (tool) phù hợp khi khách hàng yêu cầu tra cứu sản phẩm hoặc phản ánh sự cố kỹ thuật.
- Nếu không tìm thấy sản phẩm phù hợp với tiêu chí của khách hàng, phải thông báo rõ ràng, lịch sự và từ chối bịa thông tin.

## 4. OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ các sản phẩm, dịch vụ và thông tin thuộc hệ sinh thái Vingroup.
- Lịch sự từ chối và giải thích rõ đối với các yêu cầu nằm ngoài phạm vi hoạt động của Vingroup.

## 5. OUTPUT CONTRACT
Tuân thủ nghiêm ngặt quy trình suy luận ReAct:
- Thought: Phân tích ý định của khách hàng và xác định bước xử lý tiếp theo.
- Action: Tên công cụ và tham số cần gọi (hoặc 'respond_directly' nếu là FAQ).
- Observation: Kết quả nhận được từ công cụ hoặc dữ liệu thực tế.
- Final Answer: Câu trả lời hoàn chỉnh, rõ ràng và đầy đủ dành cho khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích intent độc lập và trích xuất tham số từ user_input."""
        query_lower = user_input.lower()

        # 1. Kiểm tra nhu cầu submit ticket (hỗ trợ / báo lỗi / khiếu nại)
        ticket_keywords = [
            "lỗi", "hỏng", "sự cố", "khiếu nại", "phản hồi", "hỗ trợ",
            "xử lý gấp", "nghiêm trọng", "phàn nàn", "ẩm mốc", "ticket", "báo lỗi"
        ]
        needs_ticket = any(k in query_lower for k in ticket_keywords)

        # 2. Kiểm tra nhu cầu tra cứu danh mục sản phẩm (catalog)
        catalog_intent_triggers = [
            "xem", "tìm", "mua", "tham khảo", "báo giá", "bảng giá", "dưới",
            "bao nhiêu tiền", "có xe nào", "có resort nào", "có khách sạn nào", "catalog"
        ]
        product_keywords = ["xe", "vinfast", "resort", "vinpearl", "phòng", "du lịch", "ô tô", "vf"]
        needs_catalog = any(k in query_lower for k in catalog_intent_triggers) and any(k in query_lower for k in product_keywords)

        # 3. Phân loại FAQ
        is_faq = (not needs_catalog and not needs_ticket)

        # Trích xuất tham số catalog
        catalog_params = {}
        if needs_catalog:
            if any(k in query_lower for k in ["resort", "du lịch", "du_lich", "nghỉ dưỡng", "phú quốc", "nha trang", "landmark 81", "khách sạn", "phòng"]):
                category = "du_lich"
            else:
                category = "xe_dien"

            max_price = 999999999999
            price_match = re.search(r"(?:dưới|tối đa|không quá|<)\s*(\d+(?:[.,]\d+)?)\s*(triệu|tỷ|tr|trieu|ty|đ|vnd|vnđ)?", query_lower)
            if price_match:
                val_str = price_match.group(1).replace(",", ".")
                val = float(val_str)
                unit = price_match.group(2) or ""
                if unit in ["tỷ", "ty"]:
                    max_price = int(val * 1_000_000_000)
                elif unit in ["triệu", "tr", "trieu"]:
                    max_price = int(val * 1_000_000)
                elif "triệu" in query_lower or "tr" in query_lower or "trieu" in query_lower:
                    max_price = int(val * 1_000_000)
                elif "tỷ" in query_lower or "ty" in query_lower:
                    max_price = int(val * 1_000_000_000)
                else:
                    max_price = int(val)

            catalog_params = {"category": category, "max_price": max_price}

        # Trích xuất tham số ticket
        ticket_params = {}
        if needs_ticket:
            name_match = re.search(
                r"(?:tên tôi là|tôi tên là|tôi tên|khách hàng)\s*:?\s*([A-ZĐÀ-Ỹa-zđà-ỹ\s]+?)(?:,|$|\.|\bxe\b|\bvà\b|\bphòng\b|\byêu cầu\b)",
                user_input,
                re.IGNORECASE
            )
            customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

            priority = "medium"
            if any(k in query_lower for k in ["nghiêm trọng", "gấp", "khẩn cấp", "nguy hiểm", "high"]):
                priority = "high"
            elif any(k in query_lower for k in ["thấp", "không gấp", "low"]):
                priority = "low"
            elif any(k in query_lower for k in ["trung bình", "bình thường", "medium"]):
                priority = "medium"

            issue_description = user_input
            if name_match:
                after_name = user_input[name_match.end(1):].strip(" ,.:")
                if after_name:
                    issue_description = after_name

            ticket_params = {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }

        return {
            "needs_catalog": needs_catalog,
            "catalog_params": catalog_params,
            "needs_ticket": needs_ticket,
            "ticket_params": ticket_params,
            "is_faq": is_faq
        }

    def _handle_faq(self, user_input: str) -> str:
        """Xử lý câu hỏi thường gặp (FAQ) không cần gọi tool."""
        query_lower = user_input.lower()
        if "bảo hành" in query_lower or "pin" in query_lower:
            return (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm (hoặc 200.000 km tuỳ điều kiện nào đến trước) "
                "đối với các dòng xe VF 5 Plus, VF 8, VF 9 và chính sách bảo hành tối ưu cho các dòng xe khác. "
                "Trong thời gian bảo hành, pin sẽ được sửa chữa hoặc thay thế miễn phí nếu dung lượng pin khả dụng sụt giảm dưới 70%."
            )
        return (
            "VinAssistant luôn sẵn sàng hỗ trợ quý khách về thông tin sản phẩm xe điện VinFast, "
            "dịch vụ nghỉ dưỡng Vinpearl cũng như tiếp nhận các yêu cầu hỗ trợ kỹ thuật."
        )

    def _synthesize_answer(self, catalog_result: Any, ticket_result: Any, intents: Dict[str, Any]) -> str:
        """Tổng hợp câu trả lời cuối cùng từ kết quả các tool đã thực thi."""
        parts = []
        if intents["needs_catalog"] and catalog_result is not None:
            if not catalog_result or len(catalog_result) == 0:
                parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu tìm kiếm của quý khách.")
            else:
                prod_lines = []
                for p in catalog_result:
                    prod_lines.append(f"- {p['name']} (Giá: {p['price_vnd']:,} VNĐ): {p.get('description', '')}")
                parts.append("Danh sách sản phẩm phù hợp tìm thấy:\n" + "\n".join(prod_lines))

        if intents["needs_ticket"] and ticket_result is not None:
            t_id = ticket_result.get("ticket_id", "")
            c_name = ticket_result.get("customer_name", "Quý khách")
            prio = ticket_result.get("priority", "medium")
            parts.append(
                f"Yêu cầu hỗ trợ của quý khách {c_name} đã được tiếp nhận thành công với mã ticket {t_id} "
                f"(mức độ ưu tiên: {prio}). Chuyên viên kỹ thuật sẽ liên hệ hỗ trợ quý khách trong thời gian sớm nhất."
            )

        return "\n\n".join(parts) if parts else "Đã hoàn thành xử lý yêu cầu của quý khách."

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        # TODO 3: Phân tích intent từ user_input
        intents = self._detect_intents(user_input)

        # Chuẩn bị hàng đợi hành động (Action Queue)
        actions_queue = []
        if intents["needs_catalog"]:
            actions_queue.append("search_product_catalog")
        if intents["needs_ticket"]:
            actions_queue.append("submit_support_ticket")

        # Trường hợp FAQ: Không cần gọi tool ngoại vi
        if not actions_queue:
            answer = self._handle_faq(user_input)
            self.trace.append({
                "iteration": 1,
                "step": "faq_response",
                "thought": "Câu hỏi thuộc dạng thông tin chung (FAQ), không cần gọi tool ngoại vi.",
                "action": "respond_directly",
                "observation": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # TODO 4: Xây dựng Agent Loop (while iteration <= self.max_iterations)
        iteration = 1
        catalog_result = None
        ticket_result = None

        while iteration <= self.max_iterations:
            if not actions_queue:
                break

            action_name = actions_queue.pop(0)

            if action_name == "search_product_catalog":
                params = intents["catalog_params"]
                catalog_result = search_product_catalog(**params)
                self.trace.append({
                    "iteration": iteration,
                    "step": "search_product_catalog",
                    "thought": f"Tra cứu danh mục sản phẩm {params['category']} với mức giá tối đa {params['max_price']:,} VNĐ.",
                    "action": {
                        "tool": "search_product_catalog",
                        "args": params
                    },
                    "observation": catalog_result
                })

            elif action_name == "submit_support_ticket":
                params = intents["ticket_params"]
                ticket_result = submit_support_ticket(**params)
                self.trace.append({
                    "iteration": iteration,
                    "step": "submit_support_ticket",
                    "thought": f"Tạo phiếu hỗ trợ cho khách hàng {params['customer_name']} với mức độ ưu tiên {params['priority']}.",
                    "action": {
                        "tool": "submit_support_ticket",
                        "args": params
                    },
                    "observation": ticket_result
                })

            # Nếu đã xử lý hết các action cần thiết trong vòng lặp này
            if not actions_queue:
                final_answer = self._synthesize_answer(catalog_result, ticket_result, intents)
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            iteration += 1

        # Max iterations reached safeguard
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
