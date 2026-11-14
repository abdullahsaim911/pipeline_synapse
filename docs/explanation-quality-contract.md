For **Synapse**, we defined a comprehensive explanation-quality framework to ensure that the system does more than identify objects or summarize a video frame. Its purpose is to produce **accurate, structurally complete, educationally meaningful, context-aware explanations that allow a learner to mentally reconstruct and understand the visual content without necessarily seeing it.**

Our agreed requirements were organized into **five explanation components, three detail levels, twelve visual categories, and a set of mandatory quality characteristics.**

# **1. The five core explanation components**

Every explanation should fulfill these five functions wherever applicable. They do not need to appear as five separate paragraphs or in a rigid order.

| Component                         | What it must accomplish                                                                                                     |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **1. Orientation**                | Identify what the visual represents, its topic, overall purpose, and visual type.                                           |
| **2. Structure**                  | Explain how elements are arranged, grouped, connected, ordered, and related spatially or logically.                         |
| **3. Salient visual information** | Extract the important labels, values, symbols, patterns, equations, components, and other meaningful details.               |
| **4. Educational meaning**        | Explain what the visual demonstrates, what can be concluded from it, and why its relationships matter.                      |
| **5. Lecture connection**         | Relate the visual to what the instructor is discussing, using transcript context without inventing unsupported information. |

The central principle was that **an explanation must describe both what exists in the visual and how its components work together to communicate knowledge.**

# **2. Mandatory explanation characteristics**

We identified the following quality requirements.

| #  | Characteristic                  | Expected behavior                                                                                            |
| -- | ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1  | Visual accuracy                 | Describe only information supported by reliable visual evidence.                                             |
| 2  | Structural completeness         | Capture essential elements and their organization rather than isolated object names.                         |
| 3  | Spatial clarity                 | Explain positions, relative placement, alignment, direction, and layout where meaningful.                    |
| 4  | Hierarchy                       | Preserve parent-child relationships, nesting, grouping, and visual levels.                                   |
| 5  | Connectivity and topology       | Describe arrows, links, connections, branches, paths, and interaction relationships.                         |
| 6  | Reading order                   | Follow the meaningful sequence of text, diagrams, equations, or procedural steps.                            |
| 7  | Mental-model reconstructability | Enable a learner to form a sufficiently accurate mental representation of the visual.                        |
| 8  | Educational relevance           | Prioritize details necessary for understanding the concept being taught.                                     |
| 9  | Contextual alignment            | Integrate relevant instructor narration without confusing speech with visual evidence.                       |
| 10 | Exact-content preservation      | Preserve critical mathematical symbols, code, labels, numbers, table values, and units.                      |
| 11 | Grounded interpretation         | Distinguish visible facts from supported interpretations and unsupported speculation.                        |
| 12 | Uncertainty awareness           | Represent unreadable, ambiguous, missing, partial, or conflicting evidence appropriately.                    |
| 13 | Mixed-visual completeness       | Handle multiple visual types and their relationships within the same frame.                                  |
| 14 | Nonredundancy                   | Avoid repeated descriptions, unnecessary detail, and processing that does not improve understanding.         |
| 15 | Accessibility                   | Produce clear, coherent, speech-friendly explanations.                                                       |
| 16 | Temporal correctness            | Do not infer motion, transitions, or changes from a single still frame without suitable supporting evidence. |
| 17 | Cross-level consistency         | Ensure Brief, Standard, and Detailed versions preserve the same essential facts.                             |

These were general requirements across the entire Synapse system, not rules for a particular snapshot.

# **3. Three explanation detail levels**

Brief

**Concise but complete on essential facts.**

Preserve the main teaching point and indispensable symbols, relationships, values, or other exact information. Brevity must not create misleading omissions.

Standard

**Default explanatory level.**

Include the essential structure, layout, important labels, units, relationships, patterns, and relevant lecture context. Provide enough explanation for educational understanding.

Detailed

**Expanded, evidence-grounded reconstruction.**

Include supported secondary elements, step-by-step structure, deeper relationships, comparisons, and conceptual interpretation without making unsupported claims.

Our rule was:

**Detail level changes explanation depth, not factual correctness.**

# **4. The twelve canonical visual categories**

Each visual category has different information-preservation requirements.

| Category | Visual type                                     | Essential explanation focus                                                 |
| -------- | ----------------------------------------------- | --------------------------------------------------------------------------- |
| CT-01    | Charts and graphs                               | Axes, scales, units, trends, data points, legends, comparisons              |
| CT-02    | Mathematics and symbolic notation               | Equations, exact symbols, operators, transformations, steps                 |
| CT-03    | Scientific and technical structures             | Components, labels, physical relationships, directionality, mechanisms      |
| CT-04    | Flowcharts, processes, and networks             | Nodes, arrows, branches, sequence, connectivity, dependencies               |
| CT-05    | Tables and matrices                             | Rows, columns, headers, values, associations, comparisons                   |
| CT-06    | Structured instructional text                   | Headings, bullets, definitions, grouping, reading order                     |
| CT-07    | Geometry and spatial constructions              | Shapes, angles, dimensions, coordinates, geometric relationships            |
| CT-08    | Programming and code                            | Exact code, syntax, indentation, control flow, dependencies                 |
| CT-09    | Software interfaces                             | UI components, hierarchy, navigation, controls, visual states               |
| CT-10    | Maps and spatial data                           | Locations, relative position, regions, routes, spatial relationships        |
| CT-11    | Observational images                            | Visible objects, features, relative position, observational evidence        |
| CT-12    | Instructional illustrations and physical scenes | Physical arrangements, annotated parts, interactions, demonstrated concepts |

The system must also support **multiple categories simultaneously**. For example, a frame may contain an equation, a scientific diagram, and explanatory text.

Category hints must guide extraction, not dictate the result. Qwen must identify meaningful visual categories even when the incoming classification is incomplete or incorrect.

# **5. What a high-quality explanation should achieve**

We placed particular emphasis on mental-model reconstruction.

Consider a scientific diagram showing a cell with labeled organelles and arrows.

[Animal Cell Diagram Generator — Labeled & Unlabeled | SciFig](https://images.openai.com/static-rsc-4/JFaNaSdBgGHFDPoSKEX-sqAfC77HVnNWprTFf2GH__TZJiGzVYvN-LxoGgIHaxjx9ta1mF0Z5nbWGv8Hac1ORSh3ZaZwj2joAN-KAm4SeySaYiJkDXd7s7s6SpqaQHoRGry4AZ6AAYW0fQGRCbGgVFrghgAGa1FqFnNmK3g_uMc?purpose=inline)

A weak explanation might say:

> This is a diagram of a cell showing several organelles.

A stronger explanation under our Synapse criteria would identify the cell, describe its main structural boundaries, explain where the nucleus and other important organelles appear, preserve the relevant labels, describe the meaning of arrows or connections when supported, and link the components to the biological concept being taught.

It should not invent an organelle's position, assume arrows represent movement without evidence, or add biological mechanisms that cannot be justified by the available visual and lecture context.

The goal is not the maximum number of words. It is **maximum useful, grounded understanding with minimum unnecessary information.**

# **6. How we divided responsibilities between Qwen and Mistral**

**Video frame + classification hints**

Visual input and non-authoritative category guidance

**Qwen3-VL — Visual evidence extraction**

Identify content, exact details, structural organization, relationships, spatial evidence, and uncertainty

**Unified structured visual evidence**

One canonical evidence representation

**Mistral — Explanation synthesis**

Combine evidence and relevant lecture context; produce an educational explanation

Transcript context

**Final Synapse explanation**

Brief / Standard / Detailed

We specifically agreed that:

- Qwen performs one structured extraction call per frame, covering relevant categories.
- Qwen should not perform unnecessary narrative reasoning or duplicate work better handled by Mistral.
- Mistral should transform the extracted evidence into a clear educational explanation, not fabricate missing visual details.
- The extraction must preserve the information necessary for Mistral to produce all three detail levels.
- Single-frame processing remains the current scope; temporal visual inference is deferred.

# **7. Our explanation acceptance criteria**

An explanation should be evaluated against these questions:

**Explanation quality checklist**

0 / 10

Does it correctly identify the visual and its educational purpose?

Does it preserve all essential components, labels, values, and symbols?

Does it explain the spatial arrangement, grouping, hierarchy, and topology?

Does it preserve meaningful reading order and logical relationships?

Can a learner reconstruct the important visual structure from the explanation?

Does it explain the concept rather than merely list objects?

Does it connect accurately with relevant lecture context?

Does it avoid unsupported facts, interpretations, and temporal claims?

Does it handle uncertainty and mixed-category visuals correctly?

Is it coherent, speech-friendly, nonredundant, and appropriate for the selected detail level?

Reset checklist

This checklist is a practical restatement of our quality requirements, not a claim that Synapse currently passes all of them.

## **Final definition we established**

**A Synapse explanation is a visually grounded, structurally complete, spatially coherent, educationally meaningful, context-aligned, uncertainty-aware, and detail-adaptive interpretation of instructional visual content.**

Its highest-level success criterion is that a learner can **understand what is shown, how it is organized, how its components relate, and what the visual teaches**, without unsupported claims or unnecessary processing.

These criteria should serve as the stable explanation contract against which we evaluate future changes to Qwen extraction, structured evidence, Mistral synthesis, and the overall pipeline.&#x20;