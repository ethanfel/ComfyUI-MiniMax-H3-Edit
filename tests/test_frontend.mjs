import assert from "node:assert/strict";
import fs from "node:fs";

const source = fs.readFileSync(new URL("../web/h3_edit_options.js", import.meta.url), "utf8");

assert.match(source, /LEGACY_OPTION_WIDGETS/);
assert.match(source, /optionsConnected/);
assert.match(source, /LEGACY_OPTION_WIDGETS\.has\(item\.name\)\) setWidgetVisible\(item, false\)/);
assert.match(source, /show_overrides/);
assert.match(source, /scene coverage \| cinematic hard cuts/);
assert.match(source, /scene coverage \| room \+ object study/);
assert.match(source, /expanded && scene/);
assert.match(source, /computeSize = \(\) => \[0, -4\]/);

console.log("H3 Edit compact-options frontend tests passed");
