# Full-Stack AI Agent with Google ADK, CopilotKit & AG-UI

A complete AI agent application with Google Search and Maps grounding capabilities, built with Google's Agent Development Kit (ADK), CopilotKit, and AG-UI protocol.

## ✨ Features

- 🔍 **Google Search Grounding** - Real-time web search with source attribution
- 🗺️ **Google Maps Grounding** - Location-based queries, directions, and place information
- 🎨 **Dynamic Theme** - Agent can change application colors
- 📝 **Proverbs Management** - Add/remove proverbs through conversation
- 🌦️ **Weather Cards** - Generative UI components for weather information
- 💬 **Real-time Chat** - Powered by AG-UI protocol and CopilotKit

## 🏗️ Architecture

This project demonstrates the **Agent-as-Tool Pattern** for integrating Google's grounding tools:

```
┌─────────────────┐    AG-UI Protocol    ┌─────────────────┐
│   Next.js UI    │◄──────────────────►│   ADK Agent      │
│   (CopilotKit)  │                    │   (Python)       │
└─────────────────┘                    └─────────────────┘
                                               │
                                               ▼
                                       ┌─────────────────┐
                                       │  Grounding      │
                                       │  - Search Agent │
                                       │  - Maps Agent   │
                                       └─────────────────┘
```

## 📋 Prerequisites

- **Node.js** 18+ 
- **Python** 3.12+
- **Google Cloud Account** with billing enabled
- **Google AI Studio API Key** - [Get it here](https://makersuite.google.com/app/apikey)
- **CopilotKit License Key** - [Get it here](https://cloud.copilotkit.ai/)
- **Package Manager** - npm, pnpm, yarn, or bun

## 🚀 Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd ag-ui-adk-app

# Install Node.js dependencies
npm install

# Install Python dependencies (creates virtual environment)
npm run install:agent
```

### 2. Set Up Google Cloud (For Grounding Features)

```bash
# Install Google Cloud SDK
brew install google-cloud-sdk  # macOS
# or download from: https://cloud.google.com/sdk/docs/install

# Initialize and authenticate
gcloud init
gcloud auth application-default login

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Google API Key
GOOGLE_API_KEY=your-google-api-key-here

# Vertex AI Configuration
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# CopilotKit License Key
NEXT_PUBLIC_COPILOT_LICENSE_KEY=your-copilotkit-license-key
```

### 4. Start Development Servers

```bash
npm run dev
```

This starts:
- **UI Server**: http://localhost:3000
- **Agent Server**: http://localhost:8000

## 🎯 Usage Examples

Try these prompts in the chat:

### Theme & UI
- "Set the theme to ocean blue"
- "Change the color to sunset orange"

### Proverbs
- "Write a proverb about AI"
- "Add a proverb about technology"
- "Remove the first proverb"

### Weather
- "Get the weather in San Francisco"
- "What's the weather like in Tokyo?"

### Search Grounding
- "What's the latest news about artificial intelligence?"
- "Search for information about climate change"
- "Tell me about the recent developments in quantum computing"

### Maps Grounding
- "Find restaurants near Central Park"
- "What hotels are in downtown Lagos?"
- "Give me directions from Ajao Estate to Gbagada, Lagos"
- "What's the distance between these locations?"

## 📁 Project Structure

```
ag-ui-adk-app/
├── agent/                    # ADK agent backend
│   ├── agent.py             # Main agent logic with grounding
│   └── requirements.txt     # Python dependencies
├── src/
│   └── app/
│       ├── layout.tsx       # CopilotKit provider setup
│       ├── page.tsx         # Main UI with agent integration
│       └── api/
│           └── copilotkit/
│               └── route.ts # CopilotKit API route
├── .env.example             # Environment variables template
└── README.md
```

## 🛠️ Available Scripts

```bash
# Development
npm run dev              # Start both UI and agent servers
npm run dev:ui           # Start only Next.js UI
npm run dev:agent        # Start only ADK agent
npm run dev:debug        # Start with debug logging

# Production
npm run build            # Build for production
npm run start            # Start production server

# Maintenance
npm run lint             # Run ESLint
npm run install:agent    # Install/reinstall Python dependencies
```

## 🔧 Key Implementation Details

### Agent-as-Tool Pattern

The agent uses dedicated sub-agents for grounding to avoid tool conflicts:

```python
# Search Agent - Handles web search
search_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='SearchAgent',
    tools=[GoogleSearchTool()],
)

# Maps Agent - Handles location queries
maps_agent = LlmAgent(
    model='projects/{project}/locations/{location}/publishers/google/models/gemini-2.5-flash',
    name='MapsAgent',
    tools=[GoogleMapsGroundingTool()],
)

# Main agent uses them as tools
proverbs_agent = LlmAgent(
    tools=[
        AgentTool(agent=search_agent),
        AgentTool(agent=maps_agent),
        # ... other tools
    ]
)
```

### Frontend Integration

The UI uses CopilotKit's hooks for agent interaction:

```tsx
const { state, setState } = useCoAgent<AgentState>({
  name: "my_agent",
  initialState: { /* ... */ }
});

useCopilotAction({
  name: "setThemeColor",
  parameters: [/* ... */],
  handler: async ({ color }) => {
    // Handle action
  }
});
```

## 🐛 Troubleshooting

### Agent Connection Issues
- Verify the agent is running on port 8000
- Check your Google API key is set correctly
- Ensure both servers started successfully

### Vertex AI Errors
- Confirm `gcloud auth application-default login` is run
- Verify your project has Vertex AI API enabled
- Check that `GOOGLE_CLOUD_PROJECT` is set correctly

### Python Import Errors
```bash
cd agent
source .venv/bin/activate
pip install -r requirements.txt
```

### Maps Grounding Not Working
- Ensure Vertex AI is properly configured
- Verify `GOOGLE_GENAI_USE_VERTEXAI=true` in your `.env`
- Check that you're using the Vertex AI model path for maps agent

## 📚 Learn More

- [ADK Documentation](https://google.github.io/adk-docs/)
- [CopilotKit Documentation](https://docs.copilotkit.ai)
- [AG-UI Protocol](https://github.com/CopilotKit/ag-ui)
- [Vertex AI Grounding](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview)

## 🤝 Contributing

Contributions are welcome! This is a great learning resource for building AI agents.

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- Google's Agent Development Kit (ADK)
- CopilotKit for agent UI
- AG-UI Protocol for real-time communication
- Vertex AI for grounding capabilities

---

**Next Steps:** 
- Add more grounding sources (Vertex AI Search)
- Implement authentication
- Deploy to Cloud Run
- Add database persistence
