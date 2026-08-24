import {app} from "/scripts/app.js";

const ENCODER_NODE = "TextEncodeH3Edit";
const OPTIONS_NODE = "H3EditOptions";
const SCENE_MODES = new Set([
    "scene coverage | canonical camera path",
    "scene coverage | cinematic hard cuts",
]);
const LEGACY_OPTION_WIDGETS = new Set([
    "reference_mode",
    "source_fit",
    "prompt_mode",
    "semantic_resolution",
    "native_reference_size",
    "quality_profile",
    "coverage_views",
    "coverage_arc_degrees",
    "coverage_direction",
    "coverage_hold_frames",
    "coverage_loop_closure",
]);
const COVERAGE_WIDGETS = new Set([
    "coverage_views",
    "coverage_arc_degrees",
    "coverage_direction",
    "coverage_hold_frames",
    "coverage_loop_closure",
]);

function widget(node, name) {
    return node?.widgets?.find((item) => item.name === name) ?? null;
}

function setWidgetVisible(item, visible) {
    if (!item) return;
    if (!Object.hasOwn(item, "_h3EditOriginalComputeSize")) {
        item._h3EditOriginalComputeSize = item.computeSize ?? null;
    }
    item.hidden = !visible;
    if (visible) {
        if (item._h3EditOriginalComputeSize) item.computeSize = item._h3EditOriginalComputeSize;
        else delete item.computeSize;
    } else {
        item.computeSize = () => [0, -4];
    }
}

function resize(node) {
    const width = Math.max(360, Number(node.size?.[0]) || 360);
    const height = Math.max(120, Number(node.computeSize?.()?.[1]) || 120);
    node.setSize?.([width, height]);
    node.graph?.setDirtyCanvas?.(true, true);
}

function optionsConnected(node) {
    return node?.inputs?.find((item) => item.name === "options")?.link != null;
}

function refreshEncoder(node) {
    const configured = optionsConnected(node);
    node._h3EditOriginalTitle ??= node.title;
    for (const item of node.widgets ?? []) {
        // These inputs stay in the backend schema so old workflows continue to
        // deserialize, but they are implementation details now. New workflows
        // configure them through H3 Edit Options instead of the encoder.
        if (LEGACY_OPTION_WIDGETS.has(item.name)) setWidgetVisible(item, false);
    }
    node.title = configured ? `${node._h3EditOriginalTitle} · Options` : node._h3EditOriginalTitle;
    resize(node);
}

function refreshOptions(node) {
    const mode = String(widget(node, "mode")?.value ?? "");
    const expanded = Boolean(widget(node, "show_overrides")?.value);
    const scene = SCENE_MODES.has(mode);
    for (const item of node.widgets ?? []) {
        if (["mode", "show_overrides"].includes(item.name)) continue;
        if (COVERAGE_WIDGETS.has(item.name)) setWidgetVisible(item, expanded && scene);
        else setWidgetVisible(item, expanded);
    }
    resize(node);
}

function wrapCreated(nodeType, refresh, {watchConnections = false, watchWidgets = []} = {}) {
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = created?.apply(this, arguments);
        setTimeout(() => {
            for (const widgetName of watchWidgets) {
                const watched = widget(this, widgetName);
                if (watched && !watched._h3EditWrapped) {
                    const callback = watched.callback;
                    watched.callback = function () {
                        const value = callback?.apply(this, arguments);
                        refresh(this._h3EditNode ?? null);
                        return value;
                    };
                    watched._h3EditNode = this;
                    watched._h3EditWrapped = true;
                }
            }
            refresh(this);
        }, 0);
        return result;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = configured?.apply(this, arguments);
        setTimeout(() => refresh(this), 0);
        return result;
    };
    if (watchConnections) {
        const connectionsChanged = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = connectionsChanged?.apply(this, arguments);
            setTimeout(() => refresh(this), 0);
            return result;
        };
    }
}

app.registerExtension({
    name:"h3_edit.compact_options",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === ENCODER_NODE) {
            wrapCreated(nodeType, refreshEncoder, {watchConnections:true});
        } else if (nodeData.name === OPTIONS_NODE) {
            wrapCreated(nodeType, refreshOptions, {watchWidgets:["mode", "show_overrides"]});
        }
    },
});
