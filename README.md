# Annotation generation
This module provides methods for generating annotations under given feature models (which can also be generated randomly) to create data for variational theorem proving.

## Generating a Feature Model
To generate a feature model use the following command:
```
python ./symbolic.py featuremodel input.json output.json
```

The input.json should contain just the variables under which the feature model is defined. For example:
```json
{"vars": ["a", "b", "c", "d", "e"]}
```

output.json will contain a feature model expression encoded in json, as well as the associated variables. For example:
```json
{"fm": {"op": 2, "children": [{"op": 3, "children": [{"op": 1, "children": [{"op": 0, "symbol": "d"}]}, {"op": 2, "children": [{"op": 0, "symbol": "c"}, {"op": 1, "children": [{"op": 0, "symbol": "a"}]}]}]}, {"op": 3, "children": [{"op": 0, "symbol": "b"}, {"op": 2, "children": [{"op": 0, "symbol": "c"}, {"op": 0, "symbol": "e"}]}]}]}, "vars": ["a", "b", "c", "d", "e"]}
```

## Generating Annotations

The command to use to generate annotations is:

```
python ./symbolic.py annotations input.json output.json
```
Annotations can be generated in a way such that the weight of all annotations under the feature model lies between some strict bounds `min` and `max`, and the mean weight of all annotations lies between a given range.
The feature model and variables should be provided as encoded above. `min` and `max` are floats, `mean` defines the range as a list of two floats within the range [`min`, `max`]. `n` defines the number of annotations to generate. For example:
```json
{"fm": {"op": 2, "children": [{"op": 3, "children": [{"op": 1, "children": [{"op": 0, "symbol": "d"}]}, {"op": 2, "children": [{"op": 0, "symbol": "c"}, {"op": 1, "children": [{"op": 0, "symbol": "a"}]}]}]}, {"op": 3, "children": [{"op": 0, "symbol": "b"}, {"op": 2, "children": [{"op": 0, "symbol": "c"}, {"op": 0, "symbol": "e"}]}]}]}, 
  "vars": ["a", "b", "c", "d", "e"], 
  "min": 0.6, "max": 0.9, 
  "mean": [0.7, 0.8], 
  "n": 40}
```

Output is stored as a list of json serialized expressions under the keyword `annotations`. The feature model, variables, and number of annotations is stored under the keywords `fm`, `vars`, and `n`, respectively.