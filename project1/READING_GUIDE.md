# Project 1: Reading and Method Guide

This guide provides starting points rather than a prescribed model list. Select sources that are directly relevant to the methods you intend to implement. In the report and `REFERENCES.md`, state what was adopted, what was changed, and why.

Before comparing reported Q3 values, identify whether a paper uses a single sequence, an alignment-derived profile, a structural template, or a pretrained representation, and check its label system and data splits. Results obtained with different inputs and evaluation data are not directly comparable.

## 1. Secondary-structure labels and task definition

1. **Kabsch W, Sander C.** *Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features*. Biopolymers, 1983, 22(12): 2577–2637. [doi:10.1002/bip.360221211](https://doi.org/10.1002/bip.360221211)
   - The paper introduces the DSSP assignment procedure, which distinguishes helices, strands, turns, and related states from backbone hydrogen bonds and geometry. It explains the structural origin of the labels; it is not a sequence-only predictor.
   - Relate the finer DSSP states to the H/E/C grouping used here and consider what information is lost by the Q3 mapping.

2. **Rost B, Sander C.** *Prediction of protein secondary structure at better than 70% accuracy*. Journal of Molecular Biology, 1993, 232(2): 584–599. [doi:10.1006/jmbi.1993.1413](https://doi.org/10.1006/jmbi.1993.1413)
   - This work illustrates the roles of local sequence windows, evolutionary profiles, and non-redundant train/test construction.
   - Alignment profiles are not supplied in this assignment, so the reported performance is not a direct baseline. The experimental ideas of local context, residue-level evaluation, and similarity control remain relevant.

3. **Jones DT.** *Protein secondary structure prediction based on position-specific scoring matrices*. Journal of Molecular Biology, 1999, 292(2): 195–202. [doi:10.1006/jmbi.1999.3091](https://doi.org/10.1006/jmbi.1999.3091)
   - PSIPRED uses PSI-BLAST position-specific scoring matrices and a two-stage neural network. It is a useful reference for distinguishing single-sequence encodings from evolutionary profiles.
   - Its evaluation was organised by structural similarity. Because the inputs and splits differ from this assignment, use it for task and evaluation concepts rather than direct score comparison.

## 2. Sequence-context models

4. **Wang S, Peng J, Ma J, Xu J.** *Protein secondary structure prediction using deep convolutional neural fields*. Scientific Reports, 2016, 6: 18962. [doi:10.1038/srep18962](https://doi.org/10.1038/srep18962)
   - Deep convolutions represent sequence context, while a structured output component models dependencies between neighbouring labels.
   - If only the convolutional component is implemented, distinguish it clearly from the complete published method and examine receptive field versus boundary errors.

5. **Heffernan R, Yang Y, Paliwal K, Zhou Y.** *Capturing non-local interactions by long short-term memory bidirectional recurrent neural networks for improving prediction of protein secondary structure, backbone angles, contact numbers and solvent accessibility*. Bioinformatics, 2017, 33(18): 2842–2849. [doi:10.1093/bioinformatics/btx218](https://doi.org/10.1093/bioinformatics/btx218)
   - This work uses bidirectional LSTMs to model sequence context and explicitly examines residues influenced by longer-range information. It is useful background for comparing local convolution with recurrent context models.
   - The paper predicts several structural properties and uses input features different from those supplied here. Separate the context hypothesis from these other differences when adapting the idea.

6. **Klausen MS, et al.** *NetSurfP-2.0: Improved prediction of protein structural features by integrated deep learning*. Proteins, 2019, 87(6): 520–527. [doi:10.1002/prot.25674](https://doi.org/10.1002/prot.25674)
   - The model combines convolution and bidirectional LSTM components and predicts several structural properties residue by residue.
   - The paper uses profile inputs and multitask outputs not supplied here. State which inputs and outputs are simplified and how that limits comparison.

7. **Singh J, et al.** *SPOT-1D-Single: improving the single-sequence-based prediction of protein secondary structure, backbone angles, solvent accessibility and half-sphere exposures using a large training set and ensembled deep learning*. Bioinformatics, 2021, 37(20): 3464–3472. [doi:10.1093/bioinformatics/btab316](https://doi.org/10.1093/bioinformatics/btab316)
   - This work focuses on prediction without multiple-sequence alignments and combines recurrent and convolutional components. Its single-sequence setting is comparatively close to the released assignment input.
   - It also uses a larger training set and model ensembles. Distinguish effects of data scale, architecture, and ensembling when drawing on its results.

## 3. Pretrained protein representations

8. **Rao R, et al.** *Evaluating protein transfer learning with TAPE*. Advances in Neural Information Processing Systems, 2019, 32: 9689–9701. [Paper page](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html)
   - TAPE compares pretrained representations on downstream tasks, including residue-level secondary-structure prediction.
   - If using a pretrained model, compare a clearly defined frozen or fine-tuned setting and check residue alignment after special-token handling.

9. **Rives A, et al.** *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*. PNAS, 2021, 118(15): e2016239118. [doi:10.1073/pnas.2016239118](https://doi.org/10.1073/pnas.2016239118)
   - The work studies structural and functional signals in large-scale protein language-model representations.
   - Record the exact pretrained model, layer or fine-tuning procedure, and positional alignment. Analyse why the representation differs rather than reporting only the final score.

10. **Høie MH, et al.** *NetSurfP-3.0: accurate and fast prediction of protein structural features by protein language models and deep learning*. Nucleic Acids Research, 2022, 50(W1): W510–W515. [doi:10.1093/nar/gkac439](https://doi.org/10.1093/nar/gkac439)
   - NetSurfP-3.0 replaces explicit alignment profiles with pretrained protein-language-model representations and retains residue-level structural prediction, illustrating how pretrained embeddings can feed a downstream sequence labeller.
   - It relies on large-scale pretraining and multitask outputs. If using a released representation, record the model version, extraction layer, and long-sequence handling rather than treating its published score as a course baseline.

11. **Lin Z, et al.** *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science, 2023, 379(6637): 1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
   - The work introduces ESM-2 representations and ESMFold, showing how large protein language models can encode information relevant to protein structure.
   - The paper addresses full three-dimensional structure rather than the Q3 task. It is relevant when using ESM-2 residue embeddings, but its structural outputs and published performance are not a direct target for this assignment.

## 4. Sequence similarity and evaluation

12. **Chen TR, Lo CH, Juan SH, Lo WC.** *The influence of dataset homology and a rigorous evaluation strategy on protein secondary structure prediction*. PLOS ONE, 2021, 16(7): e0254555. [doi:10.1371/journal.pone.0254555](https://doi.org/10.1371/journal.pone.0254555)
   - This study examines how homology within and across development, evaluation, and feature-reference data can affect secondary-structure prediction results. It provides context for the assignment's released similarity candidates and fixed cross-split audit rule.
   - Identity, coverage, structural similarity, and clustering can be defined differently across studies. State the comparison and threshold definitions when citing a result.

## 5. Questions to consider when choosing a method

These prompts are provided to support reading and method selection. Separate written answers are not required.

- Does the method primarily use local context, long-range context, or a pretrained sequence representation?
- Which core assumption changes relative to the other candidate?
- Does the source use information absent here, such as an alignment profile or structural template?
- Which elements can be reproduced faithfully at the present data and compute scale, and which require simplification?
- If code is reused, which modules were adopted and which parts were modified or independently implemented?
