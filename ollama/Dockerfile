# ollama/Dockerfile
FROM ollama/ollama:latest

# install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*


COPY init_ollama.sh /init_ollama.sh
RUN chmod +x /init_ollama.sh


ENTRYPOINT ["/init_ollama.sh"]
