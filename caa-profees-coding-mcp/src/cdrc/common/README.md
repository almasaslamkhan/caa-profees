# Medical Coding Project Common Files

This is the common repository shared among different medical coding projects. It contains the efforts for developing the evaluation framework, LLM model calls, and common utility functions shared among different projects.


## Key Components
Please follow the below conventions for the common folder structure:
- `data/`: Contains the data reader components including the base and the JSON reader.
- `evaluator/`: Contains the base evaluator runner. If any new runners are needed, they can inherit from the `BaseEvaluator` class.
- `metric/`: Contains a base class for metrics as well as shared metrics such as accuracy. Each new metric needs to inherit from `BaseMetric` class and implement at least a `calculate` method that receives all the outputs and the predicted values.
- `report_generation/`: Includes the report generators such as CSV or Excel.

## High-level Overview
The workflow of any evaluation includes:

1. **Configuration**: Users provide configurations including prompt templates, prompt variables, input/output schemas, and model specifics.
2. **Data Loading**: A data reader component loads data based on user-defined input/output schemas.
3. **Parallel Querying**: Data is sent to the model in parallel model calls to maximise efficiency.
4. **State Update**: The graph state is updated based on model outputs.
5. **Metrics Evaluation**: Metrics are evaluated against the user-defined criteria or the native evaluation framework metrics.
6. **Logging and Reporting**: Metrics and outputs are logged, and detailed reports are generated.

## How-to-use
To use the evaluation framework on new LLM-based applications, you need to:

1. Define your input and output schema using pydantic `BaseModel` to cast the input and output/gold data into their expected format.
2. Define your `DataLoader` by specifying a data path and assigning the input and output schema to them.
3. Define your model as a `CompliedStateGraph` in LangGraph and specify all the necessary runtime configs as a `RunnableConfig` (or a list of them for batch processing).
    - Note that the output of the compiled state graph should be casted to the output schema defined eariler. Thus, make sure to specify the correct state for the LangGraph model.
4. Specify the set of metrics for evaluation. If new metrics needs to be added, please use the `BaseMetric` class to specify your metrics.
5. Finally, you need to specify a report generator for outputting the results. Currently, we support a native `CSVReportGenerator`. In case you need to add your own metrics, please inherit from the `BaseReportGenerator` class.
6. Once we have all the components, we can wrap them up in a `BaseEvaluator` runner to run evaluation.

Below, we have written a minimal code for using the evaluation framework:

```
from cdrc.common.data import JsonDataReader
from cdrc.common.evaluator import BaseEvaluator
from cdrc.common.metric import Accuracy, FScore
from cdrc.common.report_generation import CSVReportGenerator


def main():
    # step 1 & 2
    data_reader = JsonDataReader(input_schema=INPUT_SCHEMA, output_schema=OUTPUT_SCHEMA)

    # step 3
    cpt_model, cpt_config = BUILD_LANGGRAPH_MODEL_AND_RUNNABLE_CONFIG()

    # step 4
    metrics = {
        "accuracy": Accuracy(label_key="output.label"),
        "f1_score": FScore(label_key="output.label"),
    }

    # step 5
    csv_report_gen = CSVReportGenerator(save_path=SAVE_PATH)

    # step 6: set the evaluator and run evaluation
    evaluator = BaseEvaluator(
        data_reader=data_reader,
        model=cpt_model,
        model_config=cpt_config,
        metrics=metrics,
        report_generator=csv_report_gen,
        batch_size=args.batch_size,
    )

    metrics_result, sample_report, summary_reports = evaluator.run(data_path=args.data_path)

    return metrics_result, sample_report, summary_reports


if __name__ == "__main__":
    main()

```