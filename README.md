# Data generation
This module provides methods for generating annotations under given feature models (which can also be generated randomly) as well as domain models to create data for variational theorem proving.

Any execution will only add to information provided, meaning that any config that is not used in a task will be preserved. This is useful if you want to generate a domain model from scratch by running `featuremodel`, `annotations`, and `lists.py` and writing output to the same input file.

## Requirements
The following python libraries are required:
[`sympy`](https://www.sympy.org/en/index.html) and [`pyapproxmc`](https://pypi.org/project/pyapproxmc/)

## Generating a Feature Model
To generate a feature model use the following command:
```
python ./symbolic.py featuremodel input.json output.json
```

`input.json` should contain just the variables under which the feature model is defined. For example:
```json
{"vars": ["a", "b", "c", "d", "e"]}
```

output.json will contain a feature model expression under the keyword `fm`. For example:
```json
{"fm": "(b|e&~c)&(d|c|a)", "vars": ["a", "b", "c", "d", "e"]}
```

## Generating Annotations

The command to use to generate annotations is:

```
python ./symbolic.py annotations input.json output.json
```
Annotations can be generated in a way such that the weight of all annotations under the feature model lies between some strict bounds `min` and `max`, and the mean weight of all annotations lies between a given range.
The feature model and variables should be provided as encoded above. `min` and `max` are floats, `mean` defines the range as a list of two floats within the range [`min`, `max`]. `n` defines the number of annotations to generate. For example:
```json
{"fm": "(b|e&~c)&(d|c|a)", 
  "vars": ["a", "b", "c", "d", "e"], 
  "min": 0.6, "max": 0.9, 
  "mean": [0.7, 0.8], 
  "n": 40}
```

Output is stored as a list of serialized expressions under the keyword `annotations`.

## Generating a Domain Model

The command to use to generate a domain model is:

```
python ./lists.py input.json output.json
```

`input.json` should contain but is not limited to the following vairables:

`vars`: variables used to define the scope of the data.

`annotations`: Annotations to use for generating a domain model.

`size`: Size of the domain model.

`elementRepetitions`: A list containing the number of times any element is expected to be repeated. 
For example a list of `[4, 5]` would create a model in which some element is repeated 4 times and another element is repeated 5 times.

`contiguousSublists`: A list of minimum lengths for a contiguous sublist in the domain model to have the same annotation on all elements.

A list of variational elements will be written to `domainmodel` where each variational element contains its integer value and the string encoded annotation.



