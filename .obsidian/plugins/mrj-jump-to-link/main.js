'use strict';

var obsidian = require('obsidian');
var state = require('@codemirror/state');
var view = require('@codemirror/view');

/******************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */

function __awaiter(thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
}

typeof SuppressedError === "function" ? SuppressedError : function (error, suppressed, message) {
    var e = new Error(message);
    return e.name = "SuppressedError", e.error = error, e.suppressed = suppressed, e;
};

class Settings {
    constructor() {
        // Defaults as in Vimium extension for browsers
        this.letters = 'sadfjklewcmpgh';
        this.jumpToAnywhereRegex = '\\b\\w{3,}\\b';
        this.lightspeedCaseSensitive = false;
        this.jumpToLinkIfOneLinkOnly = true;
        this.lightspeedJumpToStartOfWord = true;
        this.lightspeedCharacterCount = 2;
    }
}

class MarkWidget extends view.WidgetType {
    constructor(mark, type, matchedEventKey) {
        super();
        this.mark = mark;
        this.type = type;
        this.matchedEventKey = matchedEventKey;
    }
    eq(other) {
        return other.mark === this.mark && other.matchedEventKey == this.matchedEventKey;
    }
    toDOM() {
        const mark = activeDocument.createElement("span");
        mark.innerText = this.mark;
        const wrapper = activeDocument.createElement("div");
        wrapper.style.display = "inline-block";
        wrapper.style.position = "absolute";
        wrapper.classList.add('jl');
        wrapper.classList.add('jl-' + this.type);
        wrapper.classList.add('popover');
        if (this.matchedEventKey && this.mark.toUpperCase().startsWith(this.matchedEventKey.toUpperCase())) {
            wrapper.classList.add('matched');
        }
        wrapper.append(mark);
        return wrapper;
    }
    ignoreEvent() {
        return false;
    }
}

class MarkPlugin {
    constructor(_view) {
        this.links = [];
        this.matchedEventKey = undefined;
        this.links = [];
        this.matchedEventKey = undefined;
        this.decorations = view.Decoration.none;
    }
    setLinks(links) {
        this.links = links;
        this.matchedEventKey = undefined;
    }
    clean() {
        this.links = [];
        this.matchedEventKey = undefined;
    }
    filterWithEventKey(eventKey) {
        if (eventKey.length != 1)
            return;
        this.links = this.links.filter(v => {
            return v.letter.length == 2 && v.letter[0].toUpperCase() == eventKey.toUpperCase();
        });
        this.matchedEventKey = eventKey;
    }
    get visible() {
        return this.links.length > 0;
    }
    update(_update) {
        const widgets = this.links.map((x) => view.Decoration.widget({
            widget: new MarkWidget(x.letter, x.type, this.matchedEventKey),
            side: 1,
        }).range(x.index));
        this.decorations = view.Decoration.set(widgets);
    }
}

/**
 * Get only visible content
 * @param cmEditor
 * @returns Letter offset and visible content as a string
 */
function getVisibleLineText(cmEditor) {
    const scrollInfo = cmEditor.getScrollInfo();
    const { line: from } = cmEditor.coordsChar({ left: 0, top: 0 }, 'page');
    const { line: to } = cmEditor.coordsChar({ left: scrollInfo.left, top: scrollInfo.top + scrollInfo.height });
    const indOffset = cmEditor.indexFromPos({ ch: 0, line: from });
    const strs = cmEditor.getRange({ ch: 0, line: from }, { ch: 0, line: to + 1 });
    return { indOffset, strs };
}
/**
 *
 * @param alphabet - Letters which used to produce hints
 * @param numLinkHints - Count of needed links
 */
function getLinkHintLetters(alphabet, numLinkHints) {
    const alphabetUppercase = alphabet.toUpperCase();
    let prefixCount = Math.ceil((numLinkHints - alphabetUppercase.length) / (alphabetUppercase.length - 1));
    // ensure 0 <= prefixCount <= alphabet.length
    prefixCount = Math.max(prefixCount, 0);
    prefixCount = Math.min(prefixCount, alphabetUppercase.length);
    const prefixes = ['', ...Array.from(alphabetUppercase.slice(0, prefixCount))];
    const linkHintLetters = [];
    for (let i = 0; i < prefixes.length; i++) {
        const prefix = prefixes[i];
        for (let j = 0; j < alphabetUppercase.length; j++) {
            if (linkHintLetters.length < numLinkHints) {
                const letter = alphabetUppercase[j];
                if (prefix === '') {
                    if (!prefixes.contains(letter)) {
                        linkHintLetters.push(letter);
                    }
                }
                else {
                    linkHintLetters.push(prefix + letter);
                }
            }
            else {
                break;
            }
        }
    }
    return linkHintLetters;
}
function getMDHintLinks(content, offset, letters) {
    var _a;
    // expecting either [[Link]] or [[Link|Title]]
    const regExInternal = /\[\[(.+?)(\|.+?)?]]/g;
    // expecting [Title](../example.md)
    const regExMdInternal = /\[[^\[\]]+?\]\(((\.\.|\w|\d).+?)\)/g;
    // expecting [Title](file://link), [Title](https://link) or any other [Jira-123](jira://bla-bla) link
    const regExExternal = /\[[^\[\]]+?\]\((.+?:\/\/.+?)\)/g;
    // expecting http://hogehoge or https://hogehoge
    const regExUrl = /( |\n|^)(https?:\/\/[^ \n]+)/g;
    let indexes = new Set();
    let linksWithIndex = [];
    let regExResult;
    const addLinkToArray = (link) => {
        if (indexes.has(link.index))
            return;
        indexes.add(link.index);
        linksWithIndex.push(link);
    };
    while (regExResult = regExInternal.exec(content)) {
        const linkText = (_a = regExResult[1]) === null || _a === void 0 ? void 0 : _a.trim();
        addLinkToArray({ index: regExResult.index + offset, type: 'internal', linkText });
    }
    // External Link above internal, to prefer type external over interal in case of a dupe
    while (regExResult = regExExternal.exec(content)) {
        const linkText = regExResult[1];
        addLinkToArray({ index: regExResult.index + offset, type: 'external', linkText });
    }
    while (regExResult = regExMdInternal.exec(content)) {
        const linkText = regExResult[1];
        addLinkToArray({ index: regExResult.index + offset, type: 'internal', linkText });
    }
    while (regExResult = regExUrl.exec(content)) {
        const linkText = regExResult[2];
        addLinkToArray({ index: regExResult.index + offset + 1, type: 'external', linkText });
    }
    const linkHintLetters = getLinkHintLetters(letters, linksWithIndex.length);
    const linksWithLetter = [];
    linksWithIndex
        .sort((x, y) => x.index - y.index)
        .forEach((linkHint, i) => {
        linksWithLetter.push(Object.assign({ letter: linkHintLetters[i] }, linkHint));
    });
    return linksWithLetter.filter(link => link.letter);
}
function createWidgetElement(content, type) {
    const linkHintEl = activeDocument.createElement('div');
    linkHintEl.classList.add('jl');
    linkHintEl.classList.add('jl-' + type);
    linkHintEl.classList.add('popover');
    linkHintEl.innerHTML = content;
    return linkHintEl;
}
function displaySourcePopovers(cmEditor, linkKeyMap) {
    const drawWidget = (cmEditor, linkHint) => {
        const pos = cmEditor.posFromIndex(linkHint.index);
        // the fourth parameter is undocumented. it specifies where the widget should be place
        return cmEditor.addWidget(pos, createWidgetElement(linkHint.letter, linkHint.type), false, 'over');
    };
    linkKeyMap.forEach(x => drawWidget(cmEditor, x));
}

class CM6LinkProcessor {
    constructor(editor, alphabet) {
        this.getSourceLinkHints = () => {
            const { letters } = this;
            const { index, content } = this.getVisibleLines();
            return getMDHintLinks(content, index, letters);
        };
        this.cmEditor = editor;
        this.letters = alphabet;
    }
    init() {
        return this.getSourceLinkHints();
    }
    getVisibleLines() {
        var _a, _b, _c;
        const { cmEditor } = this;
        let { from, to } = cmEditor.viewport;
        // For CM6 get real visible lines top
        // @ts-ignore
        if ((_b = (_a = cmEditor.viewState) === null || _a === void 0 ? void 0 : _a.pixelViewport) === null || _b === void 0 ? void 0 : _b.top) {
            // @ts-ignore
            const pixelOffsetTop = cmEditor.viewState.pixelViewport.top;
            // @ts-ignore
            const lines = cmEditor.viewState.viewportLines;
            // @ts-ignore
            from = (_c = lines.filter(line => line.top > pixelOffsetTop)[0]) === null || _c === void 0 ? void 0 : _c.from;
        }
        const content = cmEditor.state.sliceDoc(from, to);
        return { index: from, content };
    }
}

function extractRegexpBlocks(content, offset, regexp, letters, caseSensitive) {
    const regExUrl = caseSensitive ? new RegExp(regexp, 'g') : new RegExp(regexp, 'ig');
    let linksWithIndex = [];
    let regExResult;
    while ((regExResult = regExUrl.exec(content))) {
        const linkText = regExResult[1];
        linksWithIndex.push({
            index: regExResult.index + offset,
            type: "regex",
            linkText,
        });
    }
    const linkHintLetters = getLinkHintLetters(letters, linksWithIndex.length);
    const linksWithLetter = [];
    linksWithIndex
        .sort((x, y) => x.index - y.index)
        .forEach((linkHint, i) => {
        linksWithLetter.push(Object.assign({ letter: linkHintLetters[i] }, linkHint));
    });
    return linksWithLetter.filter(link => link.letter);
}

class CM6RegexProcessor extends CM6LinkProcessor {
    constructor(editor, alphabet, regexp, caseSensitive) {
        super(editor, alphabet);
        this.regexp = regexp;
        this.caseSensitive = caseSensitive;
    }
    init() {
        const { letters, regexp } = this;
        const { index, content } = this.getVisibleLines();
        return extractRegexpBlocks(content, index, regexp, letters, this.caseSensitive);
    }
}

class LegacyRegexpProcessor {
    constructor(cmEditor, regexp, alphabet, caseSensitive) {
        this.cmEditor = cmEditor;
        this.regexp = regexp;
        this.letters = alphabet;
        this.caseSensitive = caseSensitive;
    }
    init() {
        const [content, offset] = this.getVisibleContent();
        const links = this.getLinks(content, offset);
        this.display(links);
        return links;
    }
    getVisibleContent() {
        const { cmEditor } = this;
        const { indOffset, strs } = getVisibleLineText(cmEditor);
        return [strs, indOffset];
    }
    getLinks(content, offset) {
        const { regexp, letters } = this;
        return extractRegexpBlocks(content, offset, regexp, letters, this.caseSensitive);
    }
    display(links) {
        const { cmEditor } = this;
        displaySourcePopovers(cmEditor, links);
    }
}

class LegacySourceLinkProcessor {
    constructor(editor, alphabet) {
        this.getSourceLinkHints = (cmEditor) => {
            const { letters } = this;
            const { indOffset, strs } = getVisibleLineText(cmEditor);
            return getMDHintLinks(strs, indOffset, letters);
        };
        this.cmEditor = editor;
        this.letters = alphabet;
    }
    init() {
        const { cmEditor } = this;
        const linkHints = this.getSourceLinkHints(cmEditor);
        displaySourcePopovers(cmEditor, linkHints);
        return linkHints;
    }
}

function getPreviewLinkHints(previewViewEl, letters) {
    const anchorEls = previewViewEl.querySelectorAll('a, .metadata-link-inner');
    const embedEls = previewViewEl.querySelectorAll('.internal-embed');
    const linkHints = [];
    anchorEls.forEach((anchorEl, _i) => {
        var _a;
        if (checkIsPreviewElOnScreen(previewViewEl, anchorEl)) {
            return;
        }
        const linkType = anchorEl.classList.contains('internal-link')
            ? 'internal'
            : 'external';
        const linkText = linkType === 'internal'
            ? (_a = anchorEl.dataset['href']) !== null && _a !== void 0 ? _a : anchorEl.href
            : anchorEl.href;
        let offsetParent = anchorEl.offsetParent;
        let top = anchorEl.offsetTop;
        let left = anchorEl.offsetLeft;
        while (offsetParent) {
            if (offsetParent == previewViewEl) {
                offsetParent = undefined;
            }
            else {
                top += offsetParent.offsetTop;
                left += offsetParent.offsetLeft;
                offsetParent = offsetParent.offsetParent;
            }
        }
        linkHints.push({
            linkElement: anchorEl,
            letter: '',
            linkText: linkText,
            type: linkType,
            top: top,
            left: left,
        });
    });
    embedEls.forEach((embedEl, _i) => {
        const linkText = embedEl.getAttribute('src');
        const linkEl = embedEl.querySelector('.markdown-embed-link');
        if (linkText && linkEl) {
            if (checkIsPreviewElOnScreen(previewViewEl, linkEl)) {
                return;
            }
            let offsetParent = linkEl.offsetParent;
            let top = linkEl.offsetTop;
            let left = linkEl.offsetLeft;
            while (offsetParent) {
                if (offsetParent == previewViewEl) {
                    offsetParent = undefined;
                }
                else {
                    top += offsetParent.offsetTop;
                    left += offsetParent.offsetLeft;
                    offsetParent = offsetParent.offsetParent;
                }
            }
            linkHints.push({
                linkElement: linkEl,
                letter: '',
                linkText: linkText,
                type: 'internal',
                top: top,
                left: left,
            });
        }
    });
    const sortedLinkHints = linkHints.sort((a, b) => {
        if (a.top > b.top) {
            return 1;
        }
        else if (a.top === b.top) {
            if (a.left > b.left) {
                return 1;
            }
            else if (a.left === b.left) {
                return 0;
            }
            else {
                return -1;
            }
        }
        else {
            return -1;
        }
    });
    const linkHintLetters = getLinkHintLetters(letters, sortedLinkHints.length);
    sortedLinkHints.forEach((linkHint, i) => {
        linkHint.letter = linkHintLetters[i];
    });
    return sortedLinkHints;
}
function checkIsPreviewElOnScreen(parent, el) {
    el = el.closest('[data-view-type="table"], table') || el;
    return el.offsetTop < parent.scrollTop || el.offsetTop > parent.scrollTop + parent.offsetHeight;
}
function displayPreviewPopovers(linkHints) {
    const linkHintHtmlElements = [];
    for (let linkHint of linkHints) {
        const popoverElement = linkHint.linkElement.createEl('span');
        linkHint.linkElement.style.position = 'relative';
        popoverElement.style.top = '0px';
        popoverElement.style.left = '0px';
        popoverElement.textContent = linkHint.letter;
        popoverElement.classList.add('jl');
        popoverElement.classList.add('jl-' + linkHint.type);
        popoverElement.classList.add('popover');
        linkHintHtmlElements.push(popoverElement);
    }
    return linkHintHtmlElements;
}

class PreviewLinkProcessor {
    constructor(view, alphabet) {
        this.view = view;
        this.alphabet = alphabet;
    }
    init() {
        const { view, alphabet } = this;
        const links = getPreviewLinkHints(view, alphabet);
        displayPreviewPopovers(links);
        return links;
    }
}

class LivePreviewLinkProcessor {
    constructor(view, editor, alphabet) {
        this.getSourceLinkHints = () => {
            const { alphabet } = this;
            const { index, content } = this.getVisibleLines();
            return getMDHintLinks(content, index, alphabet);
        };
        this.view = view;
        this.cmEditor = editor;
        this.alphabet = alphabet;
    }
    init() {
        const { view, alphabet } = this;
        const links = getPreviewLinkHints(view, alphabet);
        const sourceLinks = this.getSourceLinkHints();
        const linkHintLetters = getLinkHintLetters(alphabet, links.length + sourceLinks.length);
        const linksRemapped = links.map((link, idx) => (Object.assign(Object.assign({}, link), { letter: linkHintLetters[idx] }))).filter(link => link.letter);
        const sourceLinksRemapped = sourceLinks.map((link, idx) => (Object.assign(Object.assign({}, link), { letter: linkHintLetters[idx + links.length] }))).filter(link => link.letter);
        const linkHintHtmlElements = displayPreviewPopovers(linksRemapped);
        return [linksRemapped, sourceLinksRemapped, linkHintHtmlElements];
    }
    getVisibleLines() {
        var _a, _b, _c;
        const { cmEditor } = this;
        let { from, to } = cmEditor.viewport;
        // For CM6 get real visible lines top
        // @ts-ignore
        if ((_b = (_a = cmEditor.viewState) === null || _a === void 0 ? void 0 : _a.pixelViewport) === null || _b === void 0 ? void 0 : _b.top) {
            // @ts-ignore
            const pixelOffsetTop = cmEditor.viewState.pixelViewport.top;
            // @ts-ignore
            const lines = cmEditor.viewState.viewportLines;
            // @ts-ignore
            from = (_c = lines.filter(line => line.top > pixelOffsetTop)[0]) === null || _c === void 0 ? void 0 : _c.from;
        }
        const content = cmEditor.state.sliceDoc(from, to);
        return { index: from, content };
    }
}

var VIEW_MODE;
(function (VIEW_MODE) {
    VIEW_MODE[VIEW_MODE["SOURCE"] = 0] = "SOURCE";
    VIEW_MODE[VIEW_MODE["PREVIEW"] = 1] = "PREVIEW";
    VIEW_MODE[VIEW_MODE["LEGACY"] = 2] = "LEGACY";
    VIEW_MODE[VIEW_MODE["LIVE_PREVIEW"] = 3] = "LIVE_PREVIEW";
})(VIEW_MODE || (VIEW_MODE = {}));
class JumpToLink extends obsidian.Plugin {
    constructor() {
        super(...arguments);
        this.isLinkHintActive = false;
        this.prefixInfo = undefined;
        this.currentCursor = {};
        this.cursorBeforeJump = {};
        this.handleJumpToLink = () => {
            const { settings: { letters } } = this;
            const { mode, currentView } = this;
            switch (mode) {
                case VIEW_MODE.LEGACY: {
                    const cmEditor = this.cmEditor;
                    const sourceLinkHints = new LegacySourceLinkProcessor(cmEditor, letters).init();
                    this.handleActions(sourceLinkHints);
                    break;
                }
                case VIEW_MODE.LIVE_PREVIEW: {
                    const cm6Editor = this.cmEditor;
                    const previewViewEl = currentView.currentMode.editor.containerEl;
                    const [previewLinkHints, sourceLinkHints, linkHintHtmlElements] = new LivePreviewLinkProcessor(previewViewEl, cm6Editor, letters).init();
                    cm6Editor.plugin(this.markViewPlugin).setLinks(sourceLinkHints);
                    this.app.workspace.updateOptions();
                    this.handleActions([...previewLinkHints, ...sourceLinkHints], linkHintHtmlElements);
                    break;
                }
                case VIEW_MODE.PREVIEW: {
                    const previewViewEl = currentView.previewMode.containerEl.querySelector('div.markdown-preview-view');
                    const previewLinkHints = new PreviewLinkProcessor(previewViewEl, letters).init();
                    this.handleActions(previewLinkHints);
                    break;
                }
                case VIEW_MODE.SOURCE: {
                    const cm6Editor = this.cmEditor;
                    const livePreviewLinks = new CM6LinkProcessor(cm6Editor, letters).init();
                    cm6Editor.plugin(this.markViewPlugin).setLinks(livePreviewLinks);
                    this.app.workspace.updateOptions();
                    this.handleActions(livePreviewLinks);
                    break;
                }
            }
        };
        /*
        *  caseSensitive is only for lightspeed and shall not affect jumpToAnywhere, so it is true
        *  by default
        */
        this.handleJumpToRegex = (stringToSearch, caseSensitive = true) => {
            const { settings: { letters, jumpToAnywhereRegex } } = this;
            const whatToLookAt = stringToSearch || jumpToAnywhereRegex;
            const { mode } = this;
            switch (mode) {
                case VIEW_MODE.SOURCE:
                    this.handleMarkdownRegex(letters, whatToLookAt, caseSensitive);
                    break;
                case VIEW_MODE.LIVE_PREVIEW:
                    this.handleMarkdownRegex(letters, whatToLookAt, caseSensitive);
                    break;
                case VIEW_MODE.PREVIEW:
                    break;
                case VIEW_MODE.LEGACY:
                    const cmEditor = this.cmEditor;
                    const links = new LegacyRegexpProcessor(cmEditor, whatToLookAt, letters, caseSensitive).init();
                    this.handleActions(links);
                    break;
            }
        };
        this.handleMarkdownRegex = (letters, whatToLookAt, caseSensitive) => {
            const cm6Editor = this.cmEditor;
            const livePreviewLinks = new CM6RegexProcessor(cm6Editor, letters, whatToLookAt, caseSensitive).init();
            cm6Editor.plugin(this.markViewPlugin).setLinks(livePreviewLinks);
            this.app.workspace.updateOptions();
            this.handleActions(livePreviewLinks);
        };
    }
    onload() {
        return __awaiter(this, void 0, void 0, function* () {
            this.settings = (yield this.loadData()) || new Settings();
            this.addSettingTab(new SettingTab(this.app, this));
            const markViewPlugin = this.markViewPlugin = view.ViewPlugin.fromClass(MarkPlugin, {
                decorations: (v) => v.decorations
            });
            this.registerEditorExtension([markViewPlugin]);
            this.watchForSelectionChange();
            this.addCommand({
                id: 'activate-jump-to-link',
                name: 'Jump to Link',
                callback: this.action.bind(this, 'link'),
                hotkeys: [{ modifiers: ['Ctrl'], key: `'` }],
            });
            this.addCommand({
                id: "activate-jump-to-anywhere",
                name: "Jump to Anywhere Regex",
                callback: this.action.bind(this, 'regexp'),
                hotkeys: [{ modifiers: ["Ctrl"], key: ";" }],
            });
            this.addCommand({
                id: "activate-lightspeed-jump",
                name: "Lightspeed Jump",
                callback: this.action.bind(this, 'lightspeed'),
                hotkeys: [],
            });
        });
    }
    onunload() {
        console.log('unloading jump to links plugin');
    }
    action(type) {
        if (this.isLinkHintActive) {
            return;
        }
        const activeViewOfType = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView);
        if (!activeViewOfType) {
            return;
        }
        const currentView = this.currentView = activeViewOfType.leaf.view;
        const mode = this.mode = this.getMode(this.currentView);
        this.contentElement = activeViewOfType.contentEl;
        this.cursorBeforeJump = this.currentCursor;
        switch (mode) {
            case VIEW_MODE.LEGACY:
                this.cmEditor = currentView.sourceMode.cmEditor;
                break;
            case VIEW_MODE.LIVE_PREVIEW:
            case VIEW_MODE.SOURCE:
                this.cmEditor = currentView.editor.cm;
                break;
        }
        switch (type) {
            case "link":
                this.handleJumpToLink();
                return;
            case "regexp":
                this.handleJumpToRegex();
                return;
            case "lightspeed":
                this.handleLightspeedJump();
                return;
        }
    }
    getMode(currentView) {
        var _a;
        // @ts-ignore
        const isLegacy = this.app.vault.getConfig("legacyEditor");
        if (currentView.getState().mode === 'preview') {
            return VIEW_MODE.PREVIEW;
        }
        else if (isLegacy) {
            return VIEW_MODE.LEGACY;
        }
        else if (currentView.getState().mode === 'source') {
            try {
                const isLivePreview = (_a = currentView.editor.cm.state) === null || _a === void 0 ? void 0 : _a.field(obsidian.editorLivePreviewField);
                if (isLivePreview)
                    return VIEW_MODE.LIVE_PREVIEW;
            }
            catch (e) {
                console.error(e);
            }
            return VIEW_MODE.SOURCE;
        }
    }
    // adapted from: https://github.com/mrjackphil/obsidian-jump-to-link/issues/35#issuecomment-1085905668
    handleLightspeedJump() {
        // get all text color
        const { contentEl } = app.workspace.getActiveViewOfType(obsidian.MarkdownView);
        if (!contentEl) {
            return;
        }
        // this element doesn't exist in cm5/has a different class, so lightspeed will not work in cm5
        const contentContainerColor = contentEl.getElementsByClassName("cm-contentContainer");
        const originalColor = contentContainerColor[0].style.color;
        // change all text color to gray
        contentContainerColor[0].style.color = 'var(--jump-to-link-lightspeed-color)';
        const keyArray = [];
        const grabKey = (event) => {
            event.preventDefault();
            // handle Escape to reject the mode
            if (event.key === 'Escape') {
                contentEl.removeEventListener("keydown", grabKey, { capture: true });
                contentContainerColor[0].style.color = originalColor;
            }
            // test if keypress is capitalized
            if (/^[\w\S\W]$/i.test(event.key)) {
                const isCapital = event.shiftKey;
                if (isCapital) {
                    // capture uppercase
                    keyArray.push((event.key).toUpperCase());
                }
                else {
                    // capture lowercase
                    keyArray.push(event.key);
                }
            }
            // stop when length of array is equal to lightspeedCharacterCount
            if (keyArray.length === this.settings.lightspeedCharacterCount) {
                const stringToSearch = this.settings.lightspeedJumpToStartOfWord ? "\\b" + keyArray.join("") : keyArray.join("");
                this.handleJumpToRegex(stringToSearch, this.settings.lightspeedCaseSensitive);
                // removing eventListener after proceeded
                contentEl.removeEventListener("keydown", grabKey, { capture: true });
                contentContainerColor[0].style.color = originalColor;
            }
        };
        contentEl.addEventListener('keydown', grabKey, { capture: true });
    }
    handleHotkey(heldShiftKey, link) {
        if (link.linkElement) {
            const event = new MouseEvent("click", {
                bubbles: true,
                cancelable: true,
                view: window,
                metaKey: heldShiftKey,
            });
            link.linkElement.dispatchEvent(event);
        }
        else if (link.type === 'internal') {
            const file = this.app.workspace.getActiveFile();
            if (file) {
                // the second argument is for the link resolution
                this.app.workspace.openLinkText(decodeURI(link.linkText), file.path, heldShiftKey, { active: true });
            }
        }
        else if (link.type === 'external') {
            window.open(link.linkText);
        }
        else {
            const editor = this.cmEditor;
            if (editor instanceof view.EditorView) {
                const index = link.index;
                const { vimMode, anchor } = this.cursorBeforeJump;
                const useSelection = heldShiftKey || (vimMode === 'visual' || vimMode === 'visual block');
                if (useSelection && anchor !== undefined) {
                    editor.dispatch({ selection: state.EditorSelection.range(anchor, index) });
                }
                else {
                    editor.dispatch({ selection: state.EditorSelection.cursor(index) });
                }
            }
            else {
                editor.setCursor(editor.posFromIndex(link.index));
            }
        }
    }
    removePopovers(linkHintHtmlElements = []) {
        const currentView = this.contentElement;
        currentView.removeEventListener('click', () => this.removePopovers(linkHintHtmlElements));
        linkHintHtmlElements === null || linkHintHtmlElements === void 0 ? void 0 : linkHintHtmlElements.forEach(e => e.remove());
        currentView.querySelectorAll('.jl.popover').forEach(e => e.remove());
        this.prefixInfo = undefined;
        if (this.mode == VIEW_MODE.SOURCE || this.mode == VIEW_MODE.LIVE_PREVIEW) {
            this.cmEditor.plugin(this.markViewPlugin).clean();
        }
        this.app.workspace.updateOptions();
        this.isLinkHintActive = false;
    }
    removePopoversWithoutPrefixEventKey(eventKey, linkHintHtmlElements = []) {
        const currentView = this.contentElement;
        linkHintHtmlElements === null || linkHintHtmlElements === void 0 ? void 0 : linkHintHtmlElements.forEach(e => {
            if (e.innerHTML.length == 2 && e.innerHTML[0] == eventKey) {
                e.classList.add("matched");
                return;
            }
            e.remove();
        });
        currentView.querySelectorAll('.jl.popover').forEach(e => {
            if (e.innerHTML.length == 2 && e.innerHTML[0] == eventKey) {
                e.classList.add("matched");
                return;
            }
            e.remove();
        });
        if (this.mode == VIEW_MODE.SOURCE || this.mode == VIEW_MODE.LIVE_PREVIEW) {
            this.cmEditor.plugin(this.markViewPlugin).filterWithEventKey(eventKey);
        }
        this.app.workspace.updateOptions();
    }
    handleActions(linkHints, linkHintHtmlElements) {
        var _a;
        const contentElement = this.contentElement;
        if (!linkHints.length) {
            return;
        }
        const linkHintMap = {};
        linkHints.forEach(x => linkHintMap[x.letter] = x);
        const handleKeyDown = (event) => {
            var _a;
            if (['Shift', 'Control', 'CapsLock', 'ScrollLock', 'GroupNext', 'Meta'].includes(event.key)) {
                return;
            }
            const eventKey = event.key.toUpperCase();
            const prefixes = new Set(Object.keys(linkHintMap).filter(x => x.length > 1).map(x => x[0]));
            let linkHint;
            if (this.prefixInfo) {
                linkHint = linkHintMap[this.prefixInfo.prefix + eventKey];
            }
            else {
                linkHint = linkHintMap[eventKey];
                if (!linkHint && prefixes && prefixes.has(eventKey)) {
                    this.prefixInfo = { prefix: eventKey, shiftKey: event.shiftKey };
                    event.preventDefault();
                    event.stopPropagation();
                    event.stopImmediatePropagation();
                    this.removePopoversWithoutPrefixEventKey(eventKey, linkHintHtmlElements);
                    return;
                }
            }
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            const heldShiftKey = ((_a = this.prefixInfo) === null || _a === void 0 ? void 0 : _a.shiftKey) || event.shiftKey;
            linkHint && this.handleHotkey(heldShiftKey, linkHint);
            this.removePopovers(linkHintHtmlElements);
            contentElement.removeEventListener('keydown', handleKeyDown, { capture: true });
        };
        if (linkHints.length === 1 && this.settings.jumpToLinkIfOneLinkOnly) {
            const heldShiftKey = (_a = this.prefixInfo) === null || _a === void 0 ? void 0 : _a.shiftKey;
            this.handleHotkey(heldShiftKey, linkHints[0]);
            this.removePopovers(linkHintHtmlElements);
            return;
        }
        contentElement.addEventListener('click', () => this.removePopovers(linkHintHtmlElements));
        contentElement.addEventListener('keydown', handleKeyDown, { capture: true });
        this.isLinkHintActive = true;
    }
    /**
     * CodeMirror's vim automatically exits visual mode when executing a command.
     * This keeps track of selection changes so we can restore the selection.
     *
     * This is the same approach taken by the obsidian-vimrc-plugin
     */
    watchForSelectionChange() {
        const updateSelection = this.updateSelection.bind(this);
        const watchForChanges = () => {
            var _a, _b;
            const editor = (_a = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView)) === null || _a === void 0 ? void 0 : _a.editor;
            const cm = (_b = editor === null || editor === void 0 ? void 0 : editor.cm) === null || _b === void 0 ? void 0 : _b.cm;
            if (cm && !cm._handlers.cursorActivity.includes(updateSelection)) {
                cm.on("cursorActivity", updateSelection);
                this.register(() => cm.off("cursorActivity", updateSelection));
            }
        };
        this.registerEvent(this.app.workspace.on("active-leaf-change", watchForChanges));
        this.registerEvent(this.app.workspace.on("file-open", watchForChanges));
        watchForChanges();
    }
    updateSelection(editor) {
        var _a, _b;
        const anchor = (_a = editor.listSelections()[0]) === null || _a === void 0 ? void 0 : _a.anchor;
        this.currentCursor = {
            anchor: anchor ? editor.indexFromPos(anchor) : undefined,
            vimMode: (_b = editor.state.vim) === null || _b === void 0 ? void 0 : _b.mode
        };
    }
}
class SettingTab extends obsidian.PluginSettingTab {
    constructor(app, plugin) {
        super(app, plugin);
        this.plugin = plugin;
    }
    display() {
        let { containerEl } = this;
        containerEl.empty();
        containerEl.createEl('h2', { text: 'Settings for Jump To Link.' });
        new obsidian.Setting(containerEl)
            .setName('Characters used for link hints')
            .setDesc('The characters placed next to each link after enter link-hint mode.')
            .addText(cb => {
            cb.setValue(this.plugin.settings.letters)
                .onChange((value) => {
                this.plugin.settings.letters = value;
                this.plugin.saveData(this.plugin.settings);
            });
        });
        new obsidian.Setting(containerEl)
            .setName('Jump To Anywhere')
            .setDesc("Regex based navigating in editor mode")
            .addText((text) => text
            .setPlaceholder('Custom Regex')
            .setValue(this.plugin.settings.jumpToAnywhereRegex)
            .onChange((value) => __awaiter(this, void 0, void 0, function* () {
            this.plugin.settings.jumpToAnywhereRegex = value;
            yield this.plugin.saveData(this.plugin.settings);
        })));
        new obsidian.Setting(containerEl)
            .setName('Lightspeed regex case sensitivity')
            .setDesc('If enabled, the regex for matching will be case sensitive.')
            .addToggle((toggle) => {
            toggle.setValue(this.plugin.settings.lightspeedCaseSensitive)
                .onChange((state) => __awaiter(this, void 0, void 0, function* () {
                this.plugin.settings.lightspeedCaseSensitive = state;
                yield this.plugin.saveData(this.plugin.settings);
            }));
        });
        new obsidian.Setting(containerEl)
            .setName('Jump to Link If Only One Link In Page')
            .setDesc('If enabled, auto jump to link if there is only one link in page')
            .addToggle((toggle) => {
            toggle.setValue(this.plugin.settings.jumpToLinkIfOneLinkOnly)
                .onChange((state) => __awaiter(this, void 0, void 0, function* () {
                this.plugin.settings.jumpToLinkIfOneLinkOnly = state;
                yield this.plugin.saveData(this.plugin.settings);
            }));
        });
        new obsidian.Setting(containerEl)
            .setName('Lightspeed only jumps to start of words')
            .setDesc('If enabled, lightspeed jumps will only target characters occuring at the start of words.')
            .addToggle((toggle) => {
            toggle.setValue(this.plugin.settings.lightspeedJumpToStartOfWord)
                .onChange((state) => __awaiter(this, void 0, void 0, function* () {
                this.plugin.settings.lightspeedJumpToStartOfWord = state;
                yield this.plugin.saveData(this.plugin.settings);
            }));
        });
        new obsidian.Setting(containerEl)
            .setName('Number of characters for Lightspeed jump')
            .setDesc('Determines how many characters you need to type to perform a Lightspeed jump.')
            .addText((text) => (text
            .setValue(String(this.plugin.settings.lightspeedCharacterCount))
            .onChange((value) => __awaiter(this, void 0, void 0, function* () {
            const num = Number(value);
            if (!isNaN(num)) {
                this.plugin.settings.lightspeedCharacterCount = num;
                yield this.plugin.saveData(this.plugin.settings);
            }
        })).inputEl.type = "number"));
    }
}

module.exports = JumpToLink;


/* nosourcemap */