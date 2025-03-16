How the BidOrchestrator Works
The BidOrchestrator represents a dynamic approach to agent orchestration, using a marketplace model where agents compete and collaborate to handle user requests.

Core Concepts
Bidding System: Agents bid on user requests with confidence scores
Dynamic Team Formation: Creates teams when multiple agents would be beneficial
Performance Learning: Improves over time by tracking agent performance
Workflow
1. Bid Collection
When a user message arrives, the orchestrator:

Requests confidence scores from all available agents
Can run in parallel for faster response
For agents without a bid_on_task method, it estimates confidence based on:
Keyword overlap between message and agent description
Historical performance of the agent
2. Agent Selection
The orchestrator makes a key decision:

Use a single agent when:

One agent has significantly higher confidence (>0.7 and 0.2+ higher than others)
Only one agent is available
Form a team when:

Multiple agents have similar high confidence scores
No single agent is highly confident (below the team threshold)
3. Processing
For single agent:

Simply passes the request to the chosen agent
Supports streaming responses
Tracks performance metrics
For team-based processing:

Collects responses from all team members in parallel
Synthesizes responses into a coherent answer
If available, uses a dedicated synthesis agent
Otherwise, combines the highest confidence response with insights from others
4. Performance Learning
The orchestrator continuously updates performance metrics for agents:

Tracks average response time
Maintains success rates (could be updated with user feedback)
Uses this data to influence future agent selection
Configuration Parameters
Key parameters that control behavior:

min_confidence: Minimum confidence needed for an agent to handle a request
team_threshold: Confidence threshold for team formation
max_team_size: Maximum number of agents in a team
parallel_bidding: Whether to collect bids in parallel
Bidding Algorithm
The confidence calculation combines:

Base confidence score (0.1)
Word overlap between user query and agent description (up to 0.5)
Historical performance factor (up to 0.4)
This creates a balanced approach where teams form only when genuinely beneficial, rather than for every request.