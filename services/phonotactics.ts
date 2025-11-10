/**
 * @file phonotactics.ts
 * @description This file defines the building blocks of English-like syllables.
 * These sets of tokens (onsets, vowels, codas, etc.) are used by the custom tokenizer
 * to break down the training text and by the `wordValidator` to check if a generated
 * word has a plausible structure.
 */

// Onsets are the consonant sounds at the beginning of a syllable.
const SIMPLE_ONSETS = ['b', 'c', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v'];
const CLUSTER_ONSETS = ['br', 'pr', 'dr', 'gr', 'fr', 'cr', 'cl', 'pl', 'gl', 'fl', 'sl', 'sm', 'sn', 'sp', 'st', 'tr', 'spr', 'str', 'scr', 'sk', 'spl'];
export const ONSETS = new Set([...SIMPLE_ONSETS, ...CLUSTER_ONSETS]);

// Vowels are the core of a syllable.
export const VOWELS = new Set(['a', 'e', 'i', 'o', 'u']);

// Codas are the consonant sounds at the end of a syllable.
const SIMPLE_CODAS = ['m', 'n', 't', 'k', 'p', 'g', 'l', 'r'];
const CLUSTER_CODAS = ['mp', 'nd', 'nt', 'nk', 'st', 'sk', 'sp', 'rm', 'rn', 'rl', 'rp', 'rt', 'rk', 'sm', 'sh'];
export const CODAS = new Set([...SIMPLE_CODAS, ...CLUSTER_CODAS]);

// Extensions are common suffixes that can be appended to a syllable.
export const EXTENSIONS = new Set(['set', 'ish', 'let', 'der', 'kin', 'ling', 'ster', 'mer', 'ner', 'ler', 'gen', 'gle', 'ple', 'ble', 'dle']);

// Combine all possible phonotactic tokens into a single set.
const allTokensSet = new Set([
    ...SIMPLE_ONSETS, ...CLUSTER_ONSETS,
    ...Array.from(VOWELS),
    ...SIMPLE_CODAS, ...CLUSTER_CODAS,
    ...Array.from(EXTENSIONS)
]);

// Create a single array of all tokens, sorted by length in descending order.
// This is crucial for the greedy matching algorithm in the custom tokenizer,
// ensuring that "spl" is matched before "sp" or "s".
export const ALL_TOKENS = Array.from(allTokensSet).sort((a, b) => b.length - a.length || a.localeCompare(b));

// A comma-separated string of all tokens, used to pre-populate the custom tokenizer input field.
export const ALL_TOKENS_STRING = ALL_TOKENS.join(',');
