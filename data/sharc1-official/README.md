# QARC

The Question Answering with Rules and Conversation (QARC) dataset is presented as a challenging formulation of the Conversational Machine Reading task.


## Data Structure
The data is provided in JSON format in the `/json` directory. The train and dev splits are provided while the test set is withheld for future evaluation.

The data is structured as a list of utterances. Each utterance has the following attributes:
- `utterance_id`: a unique identifier for each utterance
- `tree_id`: an identifier unique to every rule text and question from which the utterance is produced
- `source_url`: the URL of the page from which the rule text was extracted
- `snippet`: the text containing the rule(s). (Referred to in our paper as `rule text`)
- `question`: the question we seek a response to
- `scenario`: additional context surrounding the rule text and question which the user may have provided
- `history`: a list of follow-up questions and their answers which occurred previous to the current utterance
- `answer`: the ground truth expected response for the current utterance
- `evidence`: a list of follow-up questions and their answers which would have to be inferred or understood from the scenario in order to provide the ground truth answer


## Negative Utterance IDs
A list of negative utterance IDs is provided for both the question and scenario negative examples in order to aid data selection for the sub-tasks mentioned in the paper. These are found in the `/negative_utterance_ids` directory.
- Negative Questions: `negative_question_utterance_ids.txt`
- Negative Scenarios: `negative_scenario_utterance_ids.txt`
