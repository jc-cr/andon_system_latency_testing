# andon_system_latency_testing

## Build

``bash 
cd .docker
docker-compose build
```

## Run
Update `TEST_TYPE` to what you want to test. 
The options are `cpu`, `gpu`, and `tpu`.



``bash
cd .docker
TEST_TYPE=tpu docker compose up data_sampler --remove-orphans --abort-on-container-exit
```