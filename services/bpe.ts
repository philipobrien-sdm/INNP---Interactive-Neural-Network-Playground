/**
 * @file bpe.ts
 * @description This file implements the Byte-Pair Encoding (BPE) tokenizer algorithm.
 * BPE is a sub-word tokenization strategy that starts with individual bytes and iteratively
 * merges the most frequent adjacent pairs into new, single tokens. This allows the model
 * to learn representations for common character sequences (like 'ing' or 'the') instead of
 * just single characters, which can be more efficient.
 */


/**
 * Counts the frequency of adjacent pairs of token IDs in a sequence.
 * @param ids - An array of numerical token IDs.
 * @returns A Map where keys are "ID1,ID2" strings and values are their counts.
 */
const getStats = (ids: number[]): Map<string, number> => {
    const counts = new Map<string, number>();
    for (let i = 0; i < ids.length - 1; i++) {
        const pair = `${ids[i]},${ids[i+1]}`;
        counts.set(pair, (counts.get(pair) || 0) + 1);
    }
    return counts;
};

/**
 * Replaces all occurrences of a specific pair of token IDs with a new token ID.
 * @param ids - The array of token IDs to process.
 * @param pair - The pair of IDs to merge, e.g., [101, 103].
 * @param idx - The new ID to replace the pair with.
 * @returns A new array of token IDs with the pair merged.
 */
const merge = (ids: number[], pair: number[], idx: number): number[] => {
    const newIds: number[] = [];
    let i = 0;
    while (i < ids.length) {
        // If we find the target pair at the current position...
        if (i < ids.length - 1 && ids[i] === pair[0] && ids[i+1] === pair[1]) {
            // ...replace it with the new merged token ID and advance the pointer by 2.
            newIds.push(idx);
            i += 2;
        } else {
            // Otherwise, just copy the current ID and advance by 1.
            newIds.push(ids[i]);
            i++;
        }
    }
    return newIds;
};


/**
 * Trains a BPE tokenizer on a given text.
 * @param text - The raw training text.
 * @param vocabSize - The target final vocabulary size.
 * @returns An object containing the learned `merges` map and the final `vocab` map.
 */
export const trainBPE = (text: string, vocabSize: number) => {
    // The first 256 tokens are reserved for the raw UTF-8 bytes.
    let numMerges = vocabSize - 256;
    if (numMerges < 0) numMerges = 0;
    
    // Start by encoding the text into a sequence of raw bytes (0-255).
    const textEncoder = new TextEncoder();
    let ids = Array.from(textEncoder.encode(text));
    
    const merges: Map<string, number> = new Map();
    // This `byteVocab` keeps track of the byte sequence for each token ID.
    // It's essential for correctly decoding tokens back into strings.
    const byteVocab: { [key: number]: number[] } = {};
    for (let i = 0; i < 256; i++) {
        byteVocab[i] = [i];
    }

    // The main training loop: perform `numMerges` merge operations.
    for (let i = 0; i < numMerges; i++) {
        const stats = getStats(ids);
        if (stats.size === 0) break; // Stop if there are no more pairs to merge.
        
        // Find the most frequent pair.
        const topPairStr = Array.from(stats.entries()).reduce((a, b) => a[1] > b[1] ? a : b)[0];
        const topPair = topPairStr.split(',').map(Number);
        
        const idx = 256 + i; // The new token ID for the merged pair.
        
        // Replace all occurrences of the top pair with the new token ID.
        ids = merge(ids, topPair, idx);
        merges.set(topPairStr, idx);
        
        // The byte sequence for the new token is the concatenation of its children's sequences.
        const [p1, p2] = topPair;
        byteVocab[idx] = [...(byteVocab[p1] || [p1]), ...(byteVocab[p2] || [p2])];
    }

    // Create the final, human-readable string vocabulary by decoding the byte sequences.
    const vocab: { [id: number]: string } = {};
    const decoder = new TextDecoder('utf-8', { fatal: false }); // non-fatal handles invalid byte sequences gracefully
    for (const idStr in byteVocab) {
        vocab[parseInt(idStr, 10)] = decoder.decode(new Uint8Array(byteVocab[idStr]));
    }


    return { merges, vocab };
};


/**
 * Encodes a text string into token IDs using a pre-trained BPE model.
 * Note: This is a simplified greedy implementation for educational purposes.
 * @param text - The raw text to encode.
 * @param merges - The learned merge rules from training.
 * @returns An array of numerical token IDs.
 */
export const encodeBPE = (text: string, merges: Map<string, number>): number[] => {
    const textEncoder = new TextEncoder();
    let ids: number[] = Array.from(textEncoder.encode(text));
    
    // Apply all the learned merge rules in order.
    for (const [pairStr, idx] of merges.entries()) {
        const pair = pairStr.split(',').map(Number);
        ids = merge(ids, pair, idx);
    }
    
    return ids;
};

/**
 * Decodes a sequence of token IDs back into a text string.
 * @param ids - The array of token IDs.
 * @param vocab - The vocabulary mapping IDs to string representations.
 * @returns The decoded string.
 */
export const decodeBPE = (ids: number[], vocab: { [key: number]: string }): string => {
    return ids.map(id => vocab[id] || '').join('');
};
