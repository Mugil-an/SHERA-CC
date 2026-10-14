
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends stress-ng iperf3 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

CMD ["sleep", "3600"]
