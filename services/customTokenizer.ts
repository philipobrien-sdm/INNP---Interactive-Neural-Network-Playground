/**
 * @file customTokenizer.ts
 * @description This file contains the logic for a custom tokenizer.
 * It uses a simple, greedy longest-match-first algorithm to encode a text string
 * based on a user-provided vocabulary.
 */

/**
 * Encodes a text string into a sequence of token IDs using a custom vocabulary.
 * This function iterates through the text and, at each position, finds the longest
 * token from the vocabulary that matches the current substring.
 * @param text - The raw text to encode.
 * @param vocab - The custom vocabulary, an array of strings sorted by length descending.
 * @param tokenToIndex - A map from token strings to their integer IDs.
 * @returns An array of numerical token IDs.
 */
export const encodeCustom = (text: string, vocab: string[], tokenToIndex: { [key: string]: number }): number[] => {
    // vocab is assumed to be sorted by length descending.
    const encoded: number[] = [];
    let i = 0;
    while (i < text.length) {
      let matchFound = false;
      // Iterate through the vocabulary (longest tokens first).
      for (const token of vocab) {
        // If the current token matches the text at the current position...
        if (text.startsWith(token, i)) {
          // ...add its ID to the output, advance the position, and break to the next position in the text.
          encoded.push(tokenToIndex[token]);
          i += token.length;
          matchFound = true;
          break;
        }
      }
      if (!matchFound) {
        // This should not happen if the vocab is constructed correctly
        // to include all single characters from the text, but as a fallback, we skip the character.
        i++;
      }
    }
    return encoded;
};
