# andon_system_latency_testing

## Build

``bash 
cd .docker
docker-compose build
```

## Run

``bash
cd .docker
docker compose up data_sampler --remove-orphans --abort-on-container-exit
```