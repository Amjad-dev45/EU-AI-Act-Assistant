"""Keep unsupported model knowledge out of final answers; no classifier call."""
import re

REFUSAL = ("I don't have information in the AI Act corpus to answer that request. "
           "I can help with questions about the document, exact quotations, and calculations.")

# Match the entire message: adding a greeting to a factual question is not a bypass.
CONVERSATION = re.compile(
    r"(?:hi|hello|hey|good morning|good afternoon|good evening|bonjour|"
    r"thanks|thank you|thank you very much|merci|ok|okay|bye|goodbye|"
    r"how are you|who are you|what can you do|how can you help(?: me)?|help)")


def enforce_scope(result, session, question):
    """Validated citations or stored RAG answers retain their existing behavior."""
    if result.get('citations') or result.get('answer_mode') == 'grounded_answer':
        return result, None
    normalized = re.sub(r'[^\w\s]', ' ', question.casefold())
    if CONVERSATION.fullmatch(' '.join(normalized.split())):
        return result, None

    # Without corpus evidence, render only actual successful calculator results.
    # A failed call, or an arbitrary tutorial alongside a calculation, is not proof.
    calculations = [output for (name, _), output in session.cache.items()
                    if name == 'calculator' and not output.get('error')
                    and type(output.get('result')) in (int, float)]
    if calculations and result.get('status') == 'answered':
        answer = '\n'.join(f"{row['expression']} = {row['result']:,}" for row in calculations)
        return {'status': 'answered', 'answer': answer, 'citations': [],
                'answer_mode': 'calculator_results'}, 'calculator_results_only'
    return {'status': 'refused', 'answer': REFUSAL, 'citations': [],
            'answer_mode': 'scope_refusal'}, 'no_supported_evidence'
