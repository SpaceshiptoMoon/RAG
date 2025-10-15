PROMPT_TEMPLATE = """
    下面有一个或许与这个问题相关的参考段落，若你觉得参考段落能和问题相关，则先总结参考段落的内容。
    若你觉得参考段落和问题无关，则使用你自己的原始知识来回答用户的问题，并且总是使用中文来进行回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    有用的回答:"""

QA_PROMPT = """
        # Role and Task Definition
        <|start|>
        Your core task is to **strictly base** your answers on the "Reference Context" provided to the user.
        **Absolutely prohibited** from using knowledge from your internal training data; if the information is insufficient or irrelevant, you must directly and clearly state: "Based on the provided materials, I cannot answer this question."
        </|end|>

        # Reference Context
        <|start|>
        The following are relevant text fragments retrieved based on the user's question (arranged from newest to oldest by relevance and time):
        {context}
        </|end|>

        # User Question
        <|start|>
        {question}
        </|end|>

        # Core Instructions and Output Requirements
        <|start|>
        Please strictly follow these steps:
        1. Information Sufficiency Judgment
           - If the context is insufficient to support a complete answer or is clearly unrelated, only output: "Based on the provided materials, I cannot answer this question."
           - If there are multiple mutually contradictory contexts, proceed to the "Conflict Handling" process (see Step 4).

        2. Answer Generation Principles
           - Fidelity: Answers must completely originate from the "Reference Context"; summarization and paraphrasing are allowed, but no additional or fabricated information not present in the context may be added.
           - Citation Labeling: After corresponding conclusions, label the source index in square brackets, for example: [1](@ref), [2,4](@ref). If multiple fragments jointly support the conclusion, merge the labels.
           - Structured: Use bullet points, hierarchical titles, and keypoint lists to ensure readability and logic.
        </|end|>

        # Conflict Handling Strategy (when mutually contradictory contexts appear)
        <|start|>
        - Step 1: Present conflicting claims and their sources in parallel.
        - Step 2: Adjudicate based on evidence weight (clarity, detail level, verifiability) > timeliness > scope of application.
        </|end|>

        # Strict Constraints
        <|start|>
        - Prohibited from making subjective inferences, common sense completions, or introducing external knowledge without basis.
        - Prohibited from extending or deducing the context without evidence.
        - Prohibited from altering key facts such as numbers, dates, units, regulatory clauses, or process steps.
        - Prohibited from outputting irrelevant content; maintain professionalism, restraint, and verifiability.
        - Prohibited from outputting irrelevant content; maintain professionalism, restraint, and verifiability.
        - Output language: Chinese
        </|end|>
"""