```markdown
# Design System Specification: TRACE-AML

## 1. Overview & Creative North Star
**The Creative North Star: "The Silent Intelligence"**
This design system is engineered for the high-stakes environment of financial surveillance and AML (Anti-Money Laundering) investigation. We are moving beyond the "generic dashboard" aesthetic toward a **High-Density Editorial** experience. 

The system treats data not as a series of rows, but as a forensic landscape. We break the "template" look by utilizing intentional asymmetry—where investigative sidebars might bleed into the edge of the screen—and a brutalist hierarchy that favors extreme contrast over decorative elements. It is an aesthetic of "Subtractive Elegance": if a pixel doesn't assist in identifying a pattern of crime, it is removed.

## 2. Colors & Surface Logic
The palette is a disciplined monochrome. To maintain a premium feel, we avoid the "flatness" of typical dark modes by using a tiered elevation system based on tonal shifts rather than lines.

### Surface Hierarchy & Nesting
Instead of traditional borders, hierarchy is established through **Tonal Layering**.
- **Base Layer:** `surface_container_lowest` (#0e0e0e) for the primary application canvas.
- **Structural Sections:** Use `surface_container_low` (#1b1b1b) to define large functional areas like sidebars or navigation rails.
- **Actionable Cards:** Place `surface_container` (#1f1f1f) or `surface_container_high` (#2a2a2a) elements on top of the lower tiers to create a "nested" depth.

### The "No-Line" Rule
Explicitly prohibit the use of `1px` solid borders for sectioning large layout areas. Boundaries must be defined through background color shifts. A dashboard widget should sit on the background as a slightly lighter "slab" of charcoal, not a boxed-in container.

### The "Glass & Ghost" Rule
For floating overlays (Command Palettes, Context Menus), use **Glassmorphism**.
- **Token:** `surface_variant` (#353535) at 70% opacity.
- **Effect:** `backdrop-blur: 12px`.
- **Ghost Border:** If a boundary is strictly required for accessibility, use a "Ghost Border"—the `outline_variant` (#474747) at 20% opacity. Never use 100% opaque, high-contrast borders.

## 3. Typography
The typographic voice is authoritative and clinical. We pair the neutrality of **Inter** with the forensic precision of **JetBrains Mono**.

| Level | Token | Font | Weight | Size | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Display** | `display-md` | Inter | 700 | 2.75rem | High-level KPI totals / Alert counts |
| **Headline** | `headline-sm` | Inter | 600 | 1.5rem | Section headers / Entity names |
| **Title** | `title-sm` | Inter | 500 | 1.0rem | Card titles / Form labels |
| **Body** | `body-md` | Inter | 400 | 0.875rem | Standard interface text |
| **Data/Log** | `label-sm` | Mono | 400 | 0.6875rem | Transaction hashes, timestamps, ID numbers |

**Editorial Strategy:** Use `primary` (#ffffff) for active data and `on_surface_variant` (#c6c6c6) for metadata. This creates a "scanning" effect where the investigator's eye naturally jumps to the most critical information strings.

## 4. Elevation & Depth
In this system, depth is a tool for focus, not decoration.

- **The Layering Principle:** Depth is achieved by "stacking" surface tiers. An investigation panel (`surface_container_low`) might contain several "evidence cards" (`surface_container_high`), creating a soft, natural lift.
- **Ambient Shadows:** For high-priority floating elements (e.g., a "Risk Score" modal), use an extra-diffused shadow.
    - **Shadow Color:** `rgba(0, 0, 0, 0.5)`
    - **Blur:** 24px - 40px. 
    - **Spread:** -10px (to keep the shadow tight and sophisticated).
- **Interactive States:** When a user hovers over a data row, do not use a border. Change the background color to `surface_bright` (#393939) to "illuminate" the row from beneath.

## 5. Components

### Input Fields & Search
- **Visuals:** Use `surface_container_lowest` for the field background with a `none` border. 
- **Focus State:** Transition the background to `surface_container_highest` and apply a `1px` border using the `primary` (#ffffff) token.
- **Typography:** Placeholder text must be `on_surface_variant` (#c6c6c6).

### Buttons
- **Primary:** `primary` (#ffffff) background with `on_primary` (#1a1c1c) text. Sharp `4px` (Token: `DEFAULT`) radius.
- **Secondary:** `surface_container_high` (#2a2a2a) background. No border.
- **Tertiary (Ghost):** No background. Text in `primary`. Underline only on hover.

### Data Tables & Lists (The Core of TRACE-AML)
- **Forbid Dividers:** Do not use horizontal lines between rows. Use the `0.5` spacing (0.1rem) to create a microscopic "gap" between rows where the background color peaks through, or simply use alternating tonal shifts (`surface_container_low` vs `surface_container`).
- **Density:** Use the `2` and `2.5` spacing tokens for cell padding to ensure maximum data density without sacrificing legibility.

### Forensic Chips
- **Status:** Small, monospace text in `label-sm`. 
- **Styling:** `outline_variant` border at 30% opacity. No fill for "Inactive"; solid `primary` fill for "Flagged/High Risk."

### Specialized Component: The "Investigation Thread"
- A vertical line component using `outline_variant` at 0.5px width to connect related financial transactions in a timeline view, creating a visual "paper trail."

## 6. Do's and Don'ts

### Do
- **Do** use `JetBrains Mono` for all alphanumeric strings like IBANs, Transaction IDs, and Timestamps.
- **Do** use negative space (the `8` and `10` spacing tokens) to separate different investigation modules.
- **Do** lean into "Stark White" accents for critical alerts. In a monochrome world, a pure white dot is as loud as a red one.

### Don't
- **Don't** use standard "Drop Shadows" on cards. If it’s not floating, it doesn’t have a shadow.
- **Don't** use 100% opaque borders to separate UI sections. Use tonal shifting of the background instead.
- **Don't** use rounded corners larger than `lg` (0.5rem). The system must feel sharp, precise, and professional. 
- **Don't** use pure blue or brand colors for links. Use `primary` (#ffffff) with a 1px underline to maintain the monochrome editorial integrity.```