<img src="https://decthings.com/logo.png" alt="decthings logo" width="33%" />

## Decthings model using Python

Use Python to create a Decthings model.

### Setup

Create a Decthings model using Python by going to the [create model page](https://app.decthings.com/models/create) on Decthings, and select Python as the language. This package is then by default installed and used within your model.

Manually, you can install this package using `pip3 install decthings-model`.

### Execute a model

When you create a model you define it as the variable `model` in your `main.py` file. There is of course also another side which executes your code and reads the `model` variable. This is handled automatically by Decthings when you run a model in the cloud, but in case you want to run a Decthings model on your own system you can use the Rust crate [decthings-model-executor](https://github.com/decthings/model-executor).
