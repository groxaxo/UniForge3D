# UniForge Studio design

The complete primary-screen concept is saved at
`/Users/op/.codex/generated_images/01a1562e-8089-7340-99bf-ad1fc5b19ce6/exec-431097fd-a752-443c-b98d-08b99e4d6249.png`.

Use a white header and input rail, cool pale gray canvas, charcoal text,
thin cool gray dividers and vermilion primary action. Inter, with 24 px bold
task heading, 13-14 px controls, 12 px helper copy. Desktop rail 370 px; header
56 px. Spacing follows 4/8/12/16/24/28 px, with 8 px controls and 12 px media
corners. Lucide outline icons, 18 px with 1.7 px strokes. No raster UI.

Primary copy: UniForge, Studio, Private GPU, Create an asset, Turn four views
into one 3D object., Reference images, Drop images or browse files, PNG, JPG
or WebP · 4 views of the same object, Try the example, Asset name, Seed,
Geometry, Texture, Encoder, SoftEq DINOv3, Generate asset, Preview, Textured,
Clay, Drag to orbit, Your assets, Download GLB.

Components: Header, ReferenceInput (four removable/reorderable thumbnails,
upload target, example loader), GenerationForm, ModelPreview (real WebGL GLB,
textured/clay modes, camera reset), JobProgress (actual seven-stage state,
cancel and errors), AssetShelf (select, download and provenance).

Functional deviations from the concept: geometry/texture/encoder are read-only
because only the tested 1024 / 2048 pipeline is offered. The figurine is the
actual generated GLB rather than concept artwork. The example is explicitly
identified as a previous original-encoder run; new jobs use SoftEq. The GPU
engine detail identifies CUDA source precision; local MLX is encoder-only.
Progress, errors, cancelling and job history extend the same tokens. Mobile
stacks the reference form, viewport and shelf with full-width primary action.
The model itself is not tinted or overlaid. The floor grid is CSS decoration
behind the transparent WebGL canvas. Motion follows reduced-motion settings.
Desktop panels fit the viewport and the input rail scrolls, with the generation
footer staying visible when a connection/error message adds height. Small laptop
viewports use one row of image thumbnails and compact read-only settings. GPU
health, asset file size and original/example provenance are deliberate functional
copy additions. The current GPU error state is shown instead of a green status.
