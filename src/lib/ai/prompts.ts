export const SYSTEM_PROMPT_BASE = `You are an English language tutor helping Hong Kong secondary students improve their speaking skills, especially for group discussions in oral exams.

Your task is to take the input text provided and create a realistic group discussion in English between four students (Candidate A, B, C, D) on the topic provided in the input text. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the discussion topic and question prompts as well as any relevant key points or interesting facts from the input text for the group discussion.

IMPORTANT: The input text may contain BOTH "Part A Group Interaction" (group discussion) prompts AND "Part B Individual Response" (individual response) prompts. You MUST extract the discussion topic and question prompts from "Part A Group Interaction" ONLY. Completely ignore any "Part B Individual Response" prompts, questions, or content — they are for a different task and should not influence the group discussion in any way.

Important: The ENTIRE dialogue (including brainstorming, scratchpad, and actual dialogue) should be written in English.`;

export const SYSTEM_PROMPT_RESPONSE = `You are an English language tutor helping Hong Kong secondary students improve their individual speaking and presentation skills for oral exams.

Your task is to take the input text provided and create a well-structured individual response in English that a student could deliver as a one-minute response to the given question. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the question and any relevant key points or interesting facts from the input text.

Important: The ENTIRE response (including brainstorming, scratchpad, and actual response) should be written in English.`;

export function buildBrainstormPrompt(text: string): { system: string; user: string } {
  const user = `Here is the input text you will be working with:

<input_text>
${text}
</input_text>

First, carefully read through the input text. The text may contain both "Part A Group Interaction" and "Part B Individual Response" sections. Extract the discussion topic and question prompts from "Part A Group Interaction" ONLY. Ignore all "Part B Individual Response" content entirely. Also identify any relevant key points or interesting facts from the Part A text accompanying the discussion topic.

Now brainstorm ideas and outline the discussion. Make sure your brainstorm follows the question prompts you identified from Part A Group Interaction only.

Express a range of well-developed ideas clearly, with elaboration and detail. Consider multiple perspectives and ensure ideas are suitable for a group discussion among 4 students.

Use natural, accurate vocabulary and expressions suitable for Hong Kong secondary students.

Provide your brainstorm as a structured output with: a summary of the topic, the question prompts found, key points to discuss, and your detailed brainstorming notes.`;

  return {
    system: SYSTEM_PROMPT_BASE,
    user,
  };
}

export function buildDialogueFromBrainstormPrompt(brainstorm: string): { system: string; user: string } {
  const user = `Here is a brainstorm for a group discussion:

<brainstorm>
${brainstorm}
</brainstorm>

Based on this brainstorm, write a realistic group discussion in English between four students (Candidate A, B, C, D).

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

  return {
    system: SYSTEM_PROMPT_BASE,
    user,
  };
}

export function buildDialogueLearningNotesPrompt(dialogueText: string, brainstorm: string): { system: string; user: string } {
  const user = `Here is a brainstorm outline and a group discussion dialogue that was generated from it:

<brainstorm>
${brainstorm}
</brainstorm>

<dialogue>
${dialogueText}
</dialogue>

Now create comprehensive learning notes for Hong Kong secondary students based on the dialogue above. The learning notes should have three sections:

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
    user,
  };
}

export function buildIdeasNotesPrompt(dialogueText: string, brainstorm: string): { system: string; user: string } {
  const user = `Here is a brainstorm outline and a group discussion dialogue that was generated from it:

<brainstorm>
${brainstorm}
</brainstorm>

<dialogue>
${dialogueText}
</dialogue>

Create the **Ideas** section of learning notes for Hong Kong secondary students based on the dialogue above.

Create a structured outline showing the main ideas discussed in the dialogue. Format this as HTML with proper structure:
- Use <strong> tags to bold main question prompts or key topics
- Use <em> tags to italicize important concepts or emphasis
- Use <br><br> for line breaks between major points
- Use bullet points (•) or numbered lists with <br> after each item
- Create clear hierarchy with indentation using &nbsp;&nbsp;&nbsp;&nbsp; for sub-points
- Reference the question prompts from the input text and show how the discussion addressed each one
- Include Traditional Chinese translations for all major points and sub-points

Write the content in a mix of English and Traditional Chinese to facilitate Hong Kong students' learning.`;

  return {
    system: SYSTEM_PROMPT_BASE,
    user,
  };
}

export function buildLanguageNotesPrompt(dialogueText: string): { system: string; user: string } {
  const user = `Here is a group discussion dialogue:

<dialogue>
${dialogueText}
</dialogue>

Create the **Language** section of learning notes for Hong Kong secondary students based on the dialogue above.

Identify 12-15 useful vocabulary words from the dialogue. For each item:
- Provide the English word/phrase
- Give the Traditional Chinese translation (繁體中文)
- Show how it was used in the dialogue with a brief example

Format this as an HTML table with proper structure:
<table>
<tr><th>English</th><th>中文</th><th>Usage Example</th></tr>
<tr><td><strong>word/phrase</strong></td><td>中文翻譯</td><td>Example sentence from dialogue</td>
</table>

Write the content in a mix of English and Traditional Chinese to facilitate Hong Kong students' learning.`;

  return {
    system: SYSTEM_PROMPT_BASE,
    user,
  };
}

export function buildStrategiesNotesPrompt(dialogueText: string): { system: string; user: string } {
  const user = `Here is a group discussion dialogue:

<dialogue>
${dialogueText}
</dialogue>

Create the **Communication Strategies** section of learning notes for Hong Kong secondary students based on the dialogue above.

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

Write the content in a mix of English and Traditional Chinese to facilitate Hong Kong students' learning.`;

  return {
    system: SYSTEM_PROMPT_BASE,
    user,
  };
}

export function buildDialoguePrompt(text: string): { system: string; user: string } {
  const user = `Here is the input text you will be working with:

<input_text>
${text}
</input_text>

First, carefully read through the input text. The text may contain both "Part A Group Interaction" and "Part B Individual Response" sections. Extract the discussion topic and question prompts from "Part A Group Interaction" ONLY. Ignore all "Part B Individual Response" content entirely. Also identify any relevant key points or interesting facts from the Part A text accompanying the discussion topic.

Now brainstorm ideas and outline the discussion. Make sure your discussion follows the question prompts you identified from Part A Group Interaction only.

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

export function buildIndividualResponsePrompt(text: string): { system: string; user: string } {
  const user = `Here is the input text you will be working with:

<input_text>
${text}
</input_text>

First, carefully read through the input text and identify the question being asked, as well as any relevant key points, interesting facts, or background information from the text.

Now brainstorm ideas and outline your individual response. Think about how to structure a clear, engaging one-minute response that directly addresses the question.

Write an individual response that:
- Directly answers the question in a clear, organized manner
- Uses a natural opening to introduce the topic
- Presents 2-3 well-developed main points with supporting details or examples
- Includes a brief, memorable closing or summary
- Is engaging and informative
- Uses natural, accurate vocabulary and expressions suitable for Hong Kong secondary students
- Uses a conversational tone — as if the student is speaking to an examiner or a small group

The response should be approximately 140-160 words (about 1 minute when spoken at a natural pace).

Design your output to be read aloud — it will be directly converted into audio.

Use 'Speaker' as the speaker identifier for all lines. Ensure the output strictly adheres to the required format: a list of objects, each with 'text' and 'speaker' fields.

Split the response into 3-5 natural speaking segments (paragraphs or logical pauses), each as a separate item in the array.`;

  const learningNotesPrompt = `Now create comprehensive learning notes for Hong Kong secondary students based on the individual response you just generated. The learning notes should have three sections:

**1. Ideas Section:**
Create a structured outline showing the main ideas covered in the response. Format this as HTML with proper structure:
- Use <strong> tags to bold main points or key topics
- Use <em> tags to italicize important concepts or emphasis
- Use <br><br> for line breaks between major points
- Use bullet points (•) or numbered lists with <br> after each item
- Create clear hierarchy with indentation using &nbsp;&nbsp;&nbsp;&nbsp; for sub-points
- Include Traditional Chinese translations for all major points and sub-points

**2. Language Section:**
Identify 8-12 useful vocabulary words and phrases from the response. For each item:
- Provide the English word/phrase
- Give the Traditional Chinese translation (繁體中文)
- Show how it was used in the response with a brief example

Format this as an HTML table with proper structure:
<table>
<tr><th>English</th><th>中文</th><th>Usage Example</th></tr>
<tr><td><strong>word/phrase</strong></td><td>中文翻譯</td><td>Example sentence from response</td>
</table>

**3. Communication Strategies Section:**
List and explain 4-6 presentation and speaking strategies that were demonstrated in the response. Format this as HTML with proper structure:
- Use <strong> tags to bold strategy names
- Use <em> tags to italicize example phrases from the response
- Use <br><br> for line breaks between different strategies
- Use <br> after each example phrase
- Include Traditional Chinese explanations

Strategies to include:
- Opening and introducing the topic (開場及引入主題)
- Organizing ideas clearly (清晰組織想法)
- Using transitions between points (使用過渡語連接論點)
- Providing supporting examples (提供支持性例子)
- Engaging the audience (吸引聽眾)
- Concluding effectively (有效總結)
- Using persuasive language (使用具說服力的語言)
- Expressing opinions confidently (自信地表達意見)

Write all learning notes content in a mix of English and Traditional Chinese to facilitate Hong Kong students' learning.`;

  return {
    system: SYSTEM_PROMPT_RESPONSE,
    user: `${user}\n\n${learningNotesPrompt}`,
  };
}

export function extractPartAText(text: string): string {
  const partBPattern = /\bPart\s*B\b[\s\S]*$/i;
  const result = text.replace(partBPattern, "").trimEnd();
  return result || text;
}

export const QUESTION_EXTRACTION_SYSTEM = `You are an assistant that extracts questions from HKDSE English Language Paper 4 exam or practice materials for Hong Kong secondary students.

The input text may contain BOTH "Part A Group Interaction" (group discussion) prompts AND "Part B Individual Response" (individual response) prompts. Your task is to extract ONLY the questions from Part B Individual Response. Ignore any Part A Group Interaction prompts, discussion topics, or group discussion questions entirely.

Return a JSON object with a "questions" array containing each distinct Part B Individual Response question found. If only one question is found, return it as a single-item array. If no clear Part B questions are found, return an empty array.`;

export function buildQuestionExtractionPrompt(text: string): string {
  return `Please read the following text and extract ONLY the questions from "Part B Individual Response". Ignore any "Part A Group Interaction" prompts or group discussion topics.

Each extracted question should be a complete, self-contained item that a student could respond to individually in a one-minute individual response.

<input_text>
${text}
</input_text>

Return a JSON object with a "questions" array. Each element should be the full text of one Part B Individual Response question. If the text contains only one Part B question, return a single-item array. Do NOT split sub-parts of the same question into separate items.`;
}
