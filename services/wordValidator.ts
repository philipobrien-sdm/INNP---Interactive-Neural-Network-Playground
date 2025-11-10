/**
 * @file wordValidator.ts
 * @description This file contains the logic for the "Good Word" validator.
 * This is a crucial part of the auto-coaching feature. It uses a set of heuristic rules
 * based on English phonotactics (the study of sound patterns) to determine if a
 * model-generated word is plausible-sounding.
 */

import { ONSETS, VOWELS, CODAS, EXTENSIONS, ALL_TOKENS } from './phonotactics';

/**
 * Greedily tokenizes a word string based on the phonotactic vocabulary.
 * This breaks down a word like "splint" into its constituent parts ["spl", "i", "n", "t"].
 * @param word The word string to tokenize.
 * @returns An array of token strings, or null if the word cannot be fully tokenized.
 */
const tokenizeWord = (word: string): string[] | null => {
    const tokens: string[] = [];
    let i = 0;
    while (i < word.length) {
        let matchFound = false;
        // The ALL_TOKENS list is pre-sorted by length, descending, ensuring longest match.
        for (const token of ALL_TOKENS) {
            if (word.startsWith(token, i)) {
                tokens.push(token);
                i += token.length;
                matchFound = true;
                break;
            }
        }
        if (!matchFound) {
            // If we can't parse the whole word with our phonotactic vocab, it's invalid.
            return null;
        }
    }
    return tokens;
};

// Create a list of all valid consonant clusters for rule checking.
const CONSONANT_CLUSTERS = [...new Set([...ONSETS, ...CODAS])].sort((a,b) => b.length - a.length);

/**
 * Parses a string of consonants to see if it can be formed by valid ONSET or CODA tokens.
 * This checks if a cluster like "mpst" is valid by seeing if it can be broken down into
 * known parts (e.g., "mp" + "st").
 * @param cluster The consonant cluster string.
 * @returns True if the cluster is valid, false otherwise.
 */
function isValidConsonantCluster(cluster: string): boolean {
    let i = 0;
    while (i < cluster.length) {
        let matchFound = false;
        // Greedily match longest known consonant tokens.
        for (const token of CONSONANT_CLUSTERS) {
            if (cluster.startsWith(token, i)) {
                i += token.length;
                matchFound = true;
                break;
            }
        }
        if (!matchFound) {
            return false; // Could not parse the entire cluster
        }
    }
    return true;
}


// The states for our syllable-parsing state machine.
type SyllableState = 'START' | 'ONSET' | 'VOWEL' | 'CODA' | 'EXTENSION';

/**
 * Checks if a word is "good" by applying a series of validation rules.
 * This is the main function used by the auto-coach.
 * @param word The word to validate.
 * @returns True if the word passes all rules, false otherwise.
 */
export const isGoodWord = (word: string): boolean => {
    // Rule 0: Basic length check.
    if (word.length < 2 || word.length > 12) {
        return false;
    }

    // Rule 3: Reject repetitive, unnatural patterns like 'rerer' or 'salalasa'.
    if (word.length >= 5) {
        for (let i = 0; i <= word.length - 5; i++) {
            // Pattern xyxyx
            if (word[i] === word[i+2] && word[i] === word[i+4] && word[i+1] === word[i+3]) {
                return false;
            }
        }
    }
    if (word.length >= 6) {
        for (let i = 0; i <= word.length - 6; i++) {
            // Pattern xyzxyz
            if (word[i] === word[i+3] && word[i+1] === word[i+4] && word[i+2] === word[i+5]) {
                return false;
            }
        }
    }
    
    // Rule 1: Must contain at least one vowel.
    const hasVowel = [...word].some(char => VOWELS.has(char));
    if (!hasVowel) {
        return false;
    }

    // Rule 2: Reject invalid consonant clusters of 3 or more.
    const isConsonant = (char: string) => !VOWELS.has(char);
    let consonantRun = '';
    // Append a vowel to the end to ensure the last run of consonants is checked.
    for (const char of word + 'a') { 
        if (isConsonant(char)) {
            consonantRun += char;
        } else {
            // When we hit a vowel, check the preceding consonant run.
            if (consonantRun.length >= 3) {
                if (!isValidConsonantCluster(consonantRun)) {
                    return false;
                }
            }
            // Reset the run.
            consonantRun = '';
        }
    }

    // Rule 4: Check if the word can be parsed into a valid syllable structure.
    const tokens = tokenizeWord(word);
    if (!tokens || tokens.length === 0) {
        // Word contains characters/sequences not in our phonotactic vocabulary.
        return false;
    }

    // --- Syllable State Machine ---
    // This state machine ensures the tokens appear in a logical order (e.g., a vowel
    // can't be followed by an onset in the same syllable). It also handles multi-syllable words.
    let state: SyllableState = 'START';

    for (const token of tokens) {
        switch (state) {
            case 'START':
                if (ONSETS.has(token)) {
                    state = 'ONSET';
                } else if (VOWELS.has(token)) {
                    state = 'VOWEL';
                } else {
                    return false; // Word must start with an Onset or a Vowel.
                }
                break;
            
            case 'ONSET':
                if (VOWELS.has(token)) {
                    state = 'VOWEL';
                } else if (ONSETS.has(token)) {
                    // Allow consonant clusters, e.g., 'pr' tokenized as 'p', 'r'
                    state = 'ONSET';
                } else {
                    return false; // An Onset must be followed by a Vowel or another Onset.
                }
                break;

            case 'VOWEL':
                if (CODAS.has(token)) {
                    state = 'CODA';
                } else if (EXTENSIONS.has(token)) {
                    state = 'EXTENSION';
                } else if (ONSETS.has(token)) { // New syllable starting with an onset
                    state = 'ONSET';
                } else if (VOWELS.has(token)) { // New syllable starting with a vowel
                    state = 'VOWEL';
                } else {
                    return false; // A Vowel can only be followed by a Coda, Extension, or a new syllable.
                }
                break;
            
            case 'CODA':
                if (EXTENSIONS.has(token)) {
                    state = 'EXTENSION';
                } else if (CODAS.has(token)) { // Allow Coda clusters (e.g., "mpst")
                    state = 'CODA';
                } else if (ONSETS.has(token)) { // New syllable
                    state = 'ONSET';
                } else if (VOWELS.has(token)) { // New syllable
                    state = 'VOWEL';
                } else {
                    return false; // A Coda can only be followed by an Extension or a new syllable.
                }
                break;
            
            case 'EXTENSION':
                // An extension must be the last part of a word.
                return false;
        }
    }

    // A word can validly end after a Vowel, a Coda, or an Extension.
    return state === 'VOWEL' || state === 'CODA' || state === 'EXTENSION';
};
