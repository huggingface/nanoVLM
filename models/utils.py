import re
import torch

# Used to check our models performance on multiple choice tasks. This can also be done in a more involved way with e.g. LLM-as-a-judge
def check_multiple_choice_with_regex(model_outputs, correct_answers):
    """
    Check if the model outputs contain the correct answer using regex.
    This function looks for the correct answer in the model outputs and returns a list of booleans indicating whether the answer was found.
    Returns:
        - A list of booleans indicating whether the correct answer was found in each model output.
    """
    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        correct_answer = correct_answer.upper()

        # Look for the answer letter at the beginning of a line or as the last word
        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses
        ]

        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break  # Exit inner loop once a match is found
        results.append(match_found)
    return results


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.
    """
    top_k = min(top_k, logits.size(-1))  # Safety : either take the all logits, or the top_k
    
    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None] # add None to make it broadcastable
        # Example:
        # logits = torch.tensor([[0.1, 0.2, 0.3], 
        #                       [0.4, 0.5, 0.6]])
        # top_k = 2
        # torch.topk(logits, top_k)[0] returns: tensor([[0.3, 0.2], 
        #                                               [0.6, 0.5]])
        # [..., -1, None] adds a new dimension to the tensor, making it broadcastable
        # print(logits.shape)
        # print(torch.topk(logits, top_k)[0][..., -1, None].shape) # None = unsqueeze
        # return tensor([[0.2]],
        #                [0.5]])
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        # Sort the logits in descending order, also sort the indices
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # print(sorted_logits)
        # print(sorted_indices)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # cummulatiuve sum to get the distribution 
        # print(cumulative_probs)
        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p # which tokens to remove 

        # Always keep the first token
        sorted_indices_to_remove[..., 0] = False
        # print(sorted_indices_to_remove)
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        # create new tensor with the same shape as logits, and fill it with the values from sorted_indices_to_remove
        # print(indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value) # thenn apply the mask to the logits

    return logits
