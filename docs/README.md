# manim-dataflow-analysis Documentation

## Cloning the project

You can clone the repository using the following command:

```bash
git clone https://github.com/BergLucas/manim-dataflow-analysis.git
```

## Setting up a development environment

### Requirements

The application requires:

- [Python](https://www.python.org/) ~= 3.11
- [Poetry](https://python-poetry.org/) ~= 2.0

### Python environment setup

You can setup the Python development environment by running the following command in the project root:

```bash
poetry install
```

### Code verification

You can use several commands to verify the quality of the code:

#### Running the tests

```bash
poetry poe test
```

#### Running the tests with coverage

```bash
poetry poe coverage  # only generate coverage data
# or
poetry poe coverage-report  # generate and display coverage data
# or
poetry poe coverage-html  # generate coverage data and create an HTML report
```

#### Formatting the code

```bash
poetry poe format
```

#### Checking code for errors

```bash
poetry poe check  # check the code and fix errors
# or
poetry poe check-nofix  # check the code but do not fix errors
# or
poetry poe check-unsafe  # check the code and do unsafe errors fix
```

#### Typechecking the code

```bash
poetry poe typecheck
```

#### Linting the code

```bash
poetry poe lint  # format, check and typecheck the code
```

#### Verify the code

```bash
poetry poe verify  # format, check, typecheck and test the code
```
