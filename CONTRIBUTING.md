# Contributing to FQDN-Model

Thank you for your interest in contributing to the FQDN-Model project! We welcome contributions from the community to make this tool even better.

## 🤝 How to Contribute

1.  **Fork the Repository**: unique fork for your changes.
2.  **Create a Branch**: `git checkout -b feature/amazing-feature` or `git checkout -b fix/critical-bug`.
3.  **Commit Changes**: Keep commits atomic and messages clear.
4.  **Test**: Ensure all tests pass.
5.  **Push**: `git push origin feature/amazing-feature`.
6.  **Open a Pull Request**: Describe your changes and link to any relevant issues.

## 🧪 Development Setup

1.  **Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Testing**:
    We use `pytest` for testing. **All new features must include tests.**
    ```bash
    pytest tests/
    ```

## 📐 Coding Standards

*   **Python Version**: 3.8+
*   **Style**: Follow PEP 8.
*   **Configuration**: Use `settings.py` for all constants. Do not hardcode magic numbers.
*   **Safety**:
    *   No `debug=True` in production code.
    *   No usage of `signal.SIGALRM` (breaks Windows compatibility).

## 🐛 Reporting Issues

Please use the GitHub Issue Tracker to report bugs. Include:
-   Version of the software
-   Steps to reproduce
-   Expected vs actual behavior
-   Logs/Tracebacks
