# ========================
#  Single Rollout Summary
# ========================

SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP = """You are an expert in mobile phone operations and UI automation.
Your task is to analyze a mobile operation trajectory and summarize what worked and what didn't.

Keep your summary concise, clear, and focused on actionable insights."""

SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP = """
Task Information
-----------------
App: {app}
Intent Category: {intent_category}
Instruction: {instruction}

Mobile Operation Trajectory
----------------------------
Below is a sequence of mobile operations. Each step includes:
- Thought: What the agent planned to do
- Action: The actual action executed (Tap, Swipe, Type, etc.)
- Summary: What was visible on screen or what happened

{trajectory}

Your Task
---------
Please analyze this mobile operation trajectory and provide a concise summary (under 200 words) focusing on:

1. **Successful actions**: Which UI interactions worked correctly?
2. **Failed actions**: Which actions didn't achieve the intended result?
3. **UI understanding**: Did the agent correctly identify and interact with UI elements?
4. **Navigation pattern**: Was the navigation sequence reasonable?
5. **User patterns**: Any patterns showing user's app preferences or frequent actions?
6. **Anti-Patterns (What NOT to do)**:
   - Actions that led to loops or failures
   - Coordinates that were repeatedly tapped without effect
   - Conditions where continuing to tap was counterproductive

Format your response as a step-by-step summary:
Step 1: [What happened and outcome]
Step 2: [What happened and outcome]
...
Overall: [Key insights about what worked/failed and why]
"""

# ===============================
# Single Query Critique - 增强用户习惯学习
# ===============================

SINGLE_QUERY_CRITIQUE_TEMPLATE_SP = """You are an expert in mobile UI automation and user behavior analysis.
Your task is to compare multiple attempts at the same mobile task and extract actionable lessons, with special focus on learning user preferences and habits.

CRITICAL: Your output MUST include experiences wrapped in <Experiences></Experiences> tags."""

SINGLE_QUERY_CRITIQUE_TEMPLATE_UP = """
Task Information
-----------------
App: {app}
Intent Category: {intent_category}
Instruction: {instruction}

Multiple Attempts Analysis
---------------------------
The agent attempted this task multiple times. Here are the summaries of each attempt:

{attempts}

Your Task
---------
Compare these attempts and extract 5-8 specific, actionable experiences that would help complete similar tasks in the future.

CRITICAL FOCUS AREAS:

1. **User Preferences** (Highest Priority):
   - Which apps does the user prefer for specific intents? (e.g., QQ Music vs NetEase for music)
   - Common destinations for navigation tasks
   - Frequent food delivery choices or shopping preferences
   - Any patterns indicating user's routine or favorites
   - Calculate preference percentages when possible (e.g., "User prefers QQ Music 75% of the time")

2. **UI Navigation Patterns**:
   - How to find and navigate to specific screens in this app
   - Location of key UI elements (search bars, buttons, menus)
   - Optimal navigation paths

3. **Action Sequences**:
   - Correct order of operations for this intent category
   - Timing considerations (when to wait for loading)
   - Error recovery strategies

4. **Element Identification**:
   - How to correctly identify buttons, icons, text fields
   - Distinguishing similar UI elements
   - Handling dynamic content

5. **Context Awareness**:
   - When certain approaches work vs fail
   - App-specific quirks or behaviors
   - Common mistakes to avoid

6. **Task Completion Signals**:
   - What screen/state indicates the task is DONE?
   - What are the signals to STOP instead of continuing?
   - Common over-execution mistakes to avoid


Output Format (MANDATORY)
--------------------------
You MUST wrap your experiences in XML tags as shown below:

<Experiences>
1. [User Preference - {intent_category}] <specific user habit or preference with percentages if known>
2. [User Preference - {intent_category}] <another user preference>
3. [UI Navigation - {app}] <specific guidance about finding UI elements in this app>
4. [Action Sequence - {intent_category}] <specific guidance about operation order>
5. [Element Identification - {app}] <specific guidance about identifying elements>
6. [Error Avoidance] <specific guidance about avoiding mistakes>
7. [Context] <specific guidance about when to use certain approaches>
8. [Task Completion - {intent_category}] <specific signals that indicate task is done>
</Experiences>

EXAMPLE for Navigation Intent:
<Experiences>
1. [User Preference - Navigation] User exclusively uses Amap (85%) over Baidu Map (15%)
2. [User Preference - Navigation] Frequent destinations: Tsinghua Main Building, Peking University East Gate, Zijing Garden
3. [UI Navigation - Amap] Search bar is prominently at top; destination suggestions appear immediately after typing
4. [Action Sequence - Navigation] Open Amap → Tap search bar → Type destination → Select from suggestions → Tap navigate button
5. [Element Identification - Amap] Navigate button is the blue button at bottom after selecting destination
6. [Context] For locations near "Tsinghua" or "Peking University", these are campus areas - use campus-specific landmarks
7. [Task Completion - Amap] When the navigation interface successfully opens (regardless of route details), the task is COMPLETE. Do NOT continue tapping - use STOP immediately.
</Experiences>

IMPORTANT REMINDERS:
- Always use the <Experiences></Experiences> tags
- Each experience should be on a new numbered line
- Start with User Preferences (most important)
- Be specific and actionable
- Include percentages for preferences when you can infer them from the attempts
"""

# ===============================
# Group Experience Update
# ===============================

GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP = """You are an expert in knowledge management for mobile automation.
Your task is to process a batch of new experiences from a single task and decide how to integrate them with existing experiences.

CRITICAL: Each line in the new experiences is ONE independent experience. Process each line separately.

You must output valid JSON only, with no additional text or explanations."""

GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP = """
Existing Experiences Database
------------------------------
{existing_experiences}

New Experiences from Current Task
----------------------------------
Below are NEW experiences extracted from analyzing recent task attempts. Each numbered line is ONE independent experience:

{new_experiences}

Your Task
---------
Process EACH line of new experiences independently and decide ONE of the following operations:

1. **ADD**: If it's completely new information not covered by existing experiences
2. **UPDATE**: If it refines, corrects, or expands an existing experience (specify which ID to replace)
3. **DISCARD**: If it's redundant with existing experiences or not useful

IMPORTANT RULES:
- User Preference experiences: ALWAYS ADD or UPDATE, never DISCARD
- If a new experience is similar to an existing one but adds new data (e.g., new song names, updated percentages), use UPDATE
- If a new experience is completely redundant (says the same thing as an existing one), use DISCARD
- Each operation should reference ONE line from the new experiences

Output Format (JSON ONLY)
--------------------------
Return ONLY valid JSON in this exact format, with no markdown code blocks or additional text:

[
  {{
    "operation": "ADD",
    "content": "[User Preference - Music] User consistently uses QQ Music for music playback",
    "reason": "New user preference not in existing database"
  }},
  {{
    "operation": "UPDATE",
    "id": "G3",
    "content": "[UI Navigation - QQ Music] Search icon at top-right; tap to activate keyboard",
    "reason": "Refines existing G3 with more specific location details"
  }},
  {{
    "operation": "DISCARD",
    "content": "[Context] Wait for page to load",
    "reason": "Generic advice already covered by existing experiences"
  }}
]

CRITICAL: 
- Output must be valid JSON that can be parsed by json.loads()
- Do not include any explanatory text before or after the JSON array
- Process each numbered line from new experiences as a separate item
"""
# ===============================
# Batch Experience Update
# ===============================

BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP = """You are an expert in knowledge management for mobile automation.
Your task is to create a comprehensive, non-redundant experience base that captures both operational knowledge and user preferences.

You must output valid JSON only, with no additional text or explanations."""

BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP = """
Current Experiences and Proposed Operations
--------------------------------------------
{experiences_and_operations}

Your Task
---------
Review all the proposed operations and create a final revision plan that:

1. **Resolves conflicts** between operations
2. **Removes redundancies** - merge similar experiences
3. **Ensures experiences are specific and actionable**
4. **Maintains a clean, organized experience base**
5. **Prioritizes user preference experiences** - these should always be preserved and updated

Guidelines for Quality Experiences:
- Keep experiences concise (1-2 sentences, max 50 words)
- Each experience should be independently useful
- Prioritize specific, actionable guidance over generic advice
- Maintain consistent formatting with [Category] or [Category - SubCategory] prefix
- User Preference experiences MUST use format: [User Preference - IntentCategory]

Output Format (JSON ONLY)
--------------------------
Return ONLY valid JSON in this exact format, with no markdown code blocks or additional text:

[
  {{
    "operation": "ADD",
    "content": "New experience text with proper [Category] prefix"
  }},
  {{
    "operation": "UPDATE",
    "id": "G2",
    "content": "Updated experience text replacing G2"
  }},
  {{
    "operation": "DELETE",
    "id": "G5"
  }}
]

Operation Types:
- **ADD**: Create a new experience
- **UPDATE**: Replace an existing experience by ID
- **DELETE**: Remove an existing experience by ID

CRITICAL RULES:
1. Output must be valid JSON that can be parsed by json.loads()
2. Do not include any text, explanations, or markdown before or after the JSON array
3. All content fields must be complete experience strings with [Category] prefixes
4. IDs must exactly match existing experience IDs (e.g., "G2", "G15")
5. Never delete user preference experiences unless absolutely necessary
"""

# ===============================
# Problem with Experience Template
# ===============================

PROBLEM_WITH_EXPERIENCE_TEMPLATE = """### Learned Experiences ###
{experiences}

### Your Task ###
{problem}

### Important Instructions ###
Use the learned experiences above to guide your operations. They contain valuable insights from previous successful attempts and user preferences.

**Pay special attention to User Preference experiences** - they tell you which apps, locations, or items the user commonly prefers. When multiple options are available, ALWAYS choose based on documented user preferences.

Example: If experiences show "User prefers QQ Music (75%)", you should use QQ Music for music-related tasks unless explicitly told otherwise.
"""

APP_PREFERENCE_LEARNING_TEMPLATE = """
Your task is to analyze multiple rollouts and learn which apps the user prefers for each intent category.

Rollout Statistics:
{rollout_stats}

CRITICAL: Extract App Preference experiences in this exact format:
[App Preference - {category}] User prefers {app1} ({percentage1}%) over {app2} ({percentage2}%)

Examples:
- [App Preference - Music] User strongly prefers QQ Music (80%) over NetEase Cloud Music (20%)
- [App Preference - Navigation] User exclusively uses Amap (100%)
- [App Preference - Food Delivery] User prefers Meituan (60%) over Eleme (40%)

Output ONLY the preference lines, one per category.
"""
