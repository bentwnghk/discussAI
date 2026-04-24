export const SYSTEM_PROMPT_BASE = `You are an English language tutor helping Hong Kong secondary students improve their speaking skills, especially for group discussions in oral exams.

Your task is to take the input text provided and create a realistic group discussion in English between four students (Candidate A, B, C, D) on the topic provided in the input text. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the discussion topic and question prompts as well as any relevant key points or interesting facts from the input text for the group discussion.

Important: The ENTIRE dialogue (including brainstorming, scratchpad, and actual dialogue) should be written in English.`;

export function buildDialoguePrompt(text: string): { system: string; user: string } {
  const user = `Here is the input text you will be working with:

<input_text>
${text}
</input_text>

First, carefully read through the input text and identify the discussion topic and question prompts, as well as any relevant key points or interesting facts from the text accompanying the discussion topic.

Now brainstorm ideas and outline the discussion. Make sure your discussion follows the question prompts you identified in the input text.

Express a range of well-developed ideas clearly, with elaboration and detail.

Model an authentic discussion and interaction among 4 students, and include all the following interaction strategies:
- Strategies for initiating a group discussion (e.g. Alright, we are here to discuss the proposal to ... | Let's begin by talking about the reasons why ...).
- Strategies for maintaining a group discussion (e.g. What do you think? | Any thoughts, Candidate C?).
- Strategies for transitioning in a group discussion (e.g. Does anyone have anything else to add? If not, shall we move on and discuss ...?).
- Strategies for responding in a group discussion (e.g. I agree. | That's an interesting suggestion, but I'm a bit worried that ... | Sorry, I disagree.).
- Strategies for rephrasing a group discussion (e.g. I see what you mean. You were saying that ...).
- Strategies for asking for clarification in a group discussion (e.g. I'm not sure if I understood you correctly. Did you mean that ...?).

Ensure every student contributes their ideas to every question prompt.

Use natural, accurate vocabulary and expressions suitable for Hong Kong secondary students.

Now write the full dialogue. Make it engaging, informative, 6-7 minutes long when spoken at a natural pace.

Use a conversational tone. Include all the above interaction strategies to extend the interaction naturally.

Use 'Candidate A', 'Candidate B', 'Candidate C', 'Candidate D' to identify the 4 speakers. Do not include any bracketed placeholders like [Candidate A] or [Candidate B].

Alternate speakers naturally, ensuring every candidate speaks 4-6 times throughout the discussion.

Design your output to be read aloud -- it will be directly converted into audio.

Assign appropriate speakers (Candidate A, Candidate B, Candidate C, Candidate D) to each line. Ensure the output strictly adheres to the required format: a list of objects, each with 'text' and 'speaker' fields.

Make the dialogue 6-7 minutes long when spoken at a natural pace (approximately 120-150 words per minute).

At the end of the dialogue, include a brief summary (1–2 sentences) by one of the candidates.`;

  const learningNotesPrompt = `Now create comprehensive learning notes for Hong Kong secondary students based on the dialogue you just generated. The learning notes should have three sections:

**1. Ideas Section:**
Create a structured outline showing the main ideas discussed in the dialogue. Format this as HTML with proper structure:
- Use <strong> tags to bold main question prompts or key topics
- Use <em> tags to italicize important concepts or emphasis
- Use <br><br> for line breaks between major points
- Use bullet points (•) or numbered lists with <br> after each item
- Create clear hierarchy with indentation using &nbsp;&nbsp;&nbsp;&nbsp; for sub-points
- Reference the question prompts from the input text and show how the discussion addressed each one
- Include Traditional Chinese translations for all major points and sub-points

**2. Language Section:**
Identify 12-15 useful vocabulary words from the dialogue. For each item:
- Provide the English word/phrase
- Give the Traditional Chinese translation (繁體中文)
- Show how it was used in the dialogue with a brief example

Format this as an HTML table with proper structure:
<table>
<tr><th>English</th><th>中文</th><th>Usage Example</th></tr>
<tr><td><strong>word/phrase</strong></td><td>中文翻譯</td><td>Example sentence from dialogue</td>
</table>

**3. Communication Strategies Section:**
List and explain 6-8 interaction strategies that were demonstrated in the dialogue. Format this as HTML with proper structure:
- Use <strong> tags to bold strategy names
- Use <em> tags to italicize example phrases from the dialogue
- Use <br><br> for line breaks between different strategies
- Use <br> after each example phrase
- Include Traditional Chinese explanations

Strategies to include:
- Initiating discussion (開始討論)
- Maintaining discussion (維持討論)
- Transitioning between topics (轉換話題)
- Responding and agreeing/disagreeing (回應及表達同意/不同意)
- Elaborating with examples (舉例說明)
- Building on others' ideas (延伸他人想法)
- Asking for clarification (要求澄清)
- Rephrasing (重新表述)
- Summarizing (總結)

Write all learning notes content in a mix of English and Traditional Chinese to facilitate Hong Kong students' learning.`;

  return {
    system: SYSTEM_PROMPT_BASE,
    user: `${user}\n\n${learningNotesPrompt}`,
  };
}
