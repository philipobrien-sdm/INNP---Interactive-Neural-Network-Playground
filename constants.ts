/**
 * @file constants.ts
 * @description This file holds constant values used across the application to ensure consistency
 * and make configuration changes easier.
 */

// FIX: Replaced a faulty import with a local definition for BpeMerges to resolve a type error.

/**
 * The default corpus of text the language model will train on.
 * This text is composed of plausible-sounding, non-English words designed to teach
 * the model the phonotactic rules of English-like syllable structures (e.g., which
 * consonant clusters are common at the beginning or end of words).
 */
export const DEFAULT_TRAINING_TEXT = "bresh glonder frasp splonder glant trunder vasp skam dromble slorbin drendle glantish plonder gloster glonker sprottle plinset framble prantlet slin cromp crin vask splet smet trish stram glomner plet flonker glinster frant drump smoodle crinter prundle sterm prish flinner part slimp grindlet drem gremp skinling smek closm flusk claster sprakin glimset plinder band frog prant drint blet glomster splamp flunt glant vlem splat slish skender flet glarnder crish plim blent closter fromp snish bren glinter drat sponder blisket vash glarlet snoster pramp plish fren glasket drinder spash crarlet splet bloster snent glinder prash flarlet clent crinder plasket voster spish drent gloster flasket prinder slent bren clasket blinder snash drender prasket vinder flet sninder slasket crinder spamp plet brinder blasket flent sh crinder prant blet sninder spasket vinder plamp drent clinder frasket plet glinder spinder brent snasket pramp vlet frinder blasket snent glinder splet prant plet blent sninder clasket drent glinder prasket snent blinder spamp vlet glinder spinder brent snasket";

/**
 * The number of consecutive epochs without improvement in loss before early stopping is triggered.
 * This prevents the model from "overfitting" or wasting time on unproductive training.
 */
export const EARLY_STOPPING_PATIENCE = 5;

/**
 * When early stopping triggers a fine-tuning phase, the learning rate will be decayed
 * until it reaches this multiplier of its original value. This allows for smaller,
 * more precise adjustments at the end of training.
 */
export const FINAL_LR_MULTIPLIER = 0.1;
