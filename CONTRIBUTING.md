# Contributing

Thank you for your interest in contributing to the Town Misinformation Contagion Simulator.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/contactmukundthiru-cyber/Misinformation-Agent-Simulation.git
   cd Misinformation-Agent-Simulation
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Quality

Before submitting changes, ensure your code passes all quality checks:

```bash
# Run tests
pytest

# Lint code
ruff check sim tests

# Format code
ruff format sim tests

# Type checking
mypy sim
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run the quality checks above
5. Commit with a clear message
6. Push to your fork and open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Keep functions focused and well-documented
- Write tests for new functionality

## Reporting Issues

When reporting bugs, please include:
- Python version
- PyTorch version and device (CPU/CUDA)
- Configuration file used
- Full error traceback
- Steps to reproduce

## Research Contributions

If you're contributing new cognitive mechanisms or theoretical models:
- Include citations to relevant literature
- Document the theoretical basis in docstrings
- Add validation tests against empirical targets
