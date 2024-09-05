# ElevAIte Inference Rayservice

This is the "`working_dir`" (i.e. functional source code) for the ElevAIte inference RayService.

## Installation

In your preferred Python environment:

```shell
pip install -r requirements.txt
```

## Run (development environment)

```shell
serve run endpoint:deployment library=transformers model_path="dslim/bert-base-NER" task="token-classification" trust_remote_code=0
```

(FIXME: the `trust_remote_code` value gets passed as a string of `0`, rather than as a boolean)

To test the server (assuming it is run with the default NER model):

```shell
curl -X 'POST'   'http://localhost:8000/infer'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{"args": ["Hello, world, this is John Smith"], "kwargs": {}}'
```

Or navigate to [localhost:8000/docs](http://localhost:8000/docs). The Ray dashboard can be found
at [localhost:8265](http://localhost:8265)



