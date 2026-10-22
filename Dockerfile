FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["python", "evaluate.py", "--model", "models/best_ppo_attn.pt", "--agents", "edf", "greedy", "fifo", "random", "--episodes", "100"]
