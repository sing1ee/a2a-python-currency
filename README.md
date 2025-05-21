# A2A Python Currency Agent

This repository contains a tutorial for implementing a Currency Agent using the A2A Python SDK.

## Available Languages

- 🇺🇸 [English](tutorial-en.md)
- 🇪🇸 [Spanish](tutorial-es.md)
- 🇮🇳 [Hindi](tutorial-hi.md)
- 🇸🇦 [Arabic](tutorial-ar.md)
- 🇩🇪 [German](tutorial-de.md)
- 🇯🇵 [Japanese](tutorial-ja.md)
- 🇨🇳 [Chinese (Simplified)](tutorial-zh.md)
- 🇫🇷 [French](tutorial-fr.md)
- 🇵🇹 [Portuguese](tutorial-pt.md)
- 🇷🇺 [Russian](tutorial-ru.md)

## Overview

This tutorial demonstrates how to create a Currency Agent using the official Google SDK [a2a-python](https://github.com/google/a2a-python). The agent is designed to handle currency conversion requests and provide real-time exchange rate information.

## Features

- Real-time currency conversion
- Exchange rate queries
- Multi-language support
- Streaming response capability
- Error handling
- Comprehensive testing

## Requirements

- [uv](https://docs.astral.sh/uv/#installation) 0.7.2
- Python 3.13+
- OpenAI/OpenRouter API key

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/sing1ee/a2a-python-currency.git
cd a2a-python-currency
```

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv add a2a-sdk uvicorn dotenv click
```

4. Set up environment variables:
```bash
echo OPENROUTER_API_KEY=your_api_key >> .env
echo OPENROUTER_BASE_URL=your_base_url >> .env
```

5. Run the server:
```bash
uv run python main.py
```

6. Run the client:
```bash
uv run python test_client.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
