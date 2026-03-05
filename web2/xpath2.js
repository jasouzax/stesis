// xpath.js

class XPathParser {
    constructor(str) {
        this.variables = {};
        this.str = this.preprocessPiping(str);
    }

    preprocessPiping(str) {
        let parts = str.split('>');
        if (parts.length <= 1) return str;
        let baseStr = parts[0];
        let blocks = parts.slice(1);
        for (let block of blocks) {
            baseStr = baseStr.replace('<', block);
        }
        return baseStr.replace(/</g, '');
    }

    tokenize() {
        let tokens = [];
        // Updated regex: Captures declarations {var=...}, strings, single-chars, and values containing math {3*u+5}
        let tokenRegex = /\([^)]*\)|\{[a-zA-Z0-9_]+=[^}]+\}|#[a-fA-F0-9]+|[a-zA-Z]{2,}|(?:\{[^}]+\}|[-+0-9.%,;])+|[a-zA-Z\[\]\|]/g;
        let match;

        while ((match = tokenRegex.exec(this.str)) !== null) {
            let val = match[0];
            if (val === '') {
                tokenRegex.lastIndex++;
                continue;
            }

            if (val.startsWith('(')) {
                continue;
            } else if (val.startsWith('{') && val.includes('=')) {
                let inner = val.slice(1, -1);
                let eqIdx = inner.indexOf('=');
                this.variables[inner.slice(0, eqIdx)] = inner.slice(eqIdx + 1);
                continue;
            } else if (val.startsWith('#') || /^[a-zA-Z]{2,}$/.test(val)) {
                tokens.push({ type: 'string', value: val });
            } else if (/^[a-zA-Z\[\]\|]$/.test(val)) {
                tokens.push({ type: 'command', value: val });
            } else {
                tokens.push({ type: 'value', value: val });
            }
        }

        // Implicitly inject '|' into any top-level block that doesn't have one
        let outTokens = [];
        let i = 0;
        while (i < tokens.length) {
            if (tokens[i].value === '[') {
                let blockStart = i;
                let blockDepth = 0;
                let blockEnd = i;
                for (let j = i; j < tokens.length; j++) {
                    if (tokens[j].value === '[') blockDepth++;
                    else if (tokens[j].value === ']') blockDepth--;
                    if (blockDepth === 0) { blockEnd = j; break; }
                }
                if (blockDepth !== 0) blockEnd = tokens.length - 1;

                let blockTokens = tokens.slice(blockStart, blockEnd + 1);
                let hasPipe = blockTokens.some(t => t.value === '|');
                if (!hasPipe) {
                    let firstCloseIdx = blockTokens.findIndex(t => t.value === ']');
                    if (firstCloseIdx !== -1) {
                        blockTokens.splice(firstCloseIdx, 0, { type: 'command', value: '|' });
                    }
                }
                outTokens.push(...blockTokens);
                i = blockEnd + 1;
            } else {
                outTokens.push(tokens[i]);
                i++;
            }
        }
        return outTokens;
    }
}

const COMMAND_ARGS = {
    'M': ['x', 'y'], 'm': ['x', 'y'],
    'L': ['x', 'y'], 'l': ['x', 'y'],
    'H': ['x'], 'h': ['x'],
    'V': ['y'], 'v': ['y'],
    'C': ['x', 'y', 'x', 'y', 'x', 'y'], 'c': ['x', 'y', 'x', 'y', 'x', 'y'],
    'S': ['x', 'y', 'x', 'y'], 's': ['x', 'y', 'x', 'y'],
    'Q': ['x', 'y', 'x', 'y'], 'q': ['x', 'y', 'x', 'y'],
    'T': ['x', 'y'], 't': ['x', 'y'],
    'A': ['x', 'y', 'none', 'none', 'none', 'x', 'y'], 'a': ['x', 'y', 'none', 'none', 'none', 'x', 'y'],
    'Z': [], 'z': [],
    'O': ['x', 'y'], 'o': ['x', 'y']
};

class XPath {
    constructor(str) {
        this.str = str;
        this.parser = new XPathParser(str);
        this.tokens = this.parser.tokenize();
        this.paths = [];
        this.hasRelativeX = false;
        this.hasRelativeY = false;
        this.fixedSize = null;
        this.sizeRatio = null;
        this.parse();
    }
    parsePO(valStr) {
        let terms = [];
        // Extract multipliers, variables, and literal strings dynamically
        let regex = /\{([0-9.]+)?([a-zA-Z0-9_]+)\}|([-+0-9.%,;]+)/g;
        let m;
        while ((m = regex.exec(valStr)) !== null) {
            if (m[2]) {
                terms.push({ type: 'var', coeff: m[1] !== undefined ? parseFloat(m[1]) : 1, name: m[2] });
            } else if (m[3]) {
                terms.push({ type: 'literal', val: m[3] });
            }
        }

        if (terms.length === 0) return [[{ p: 0, o: 0 }]];

        // Evaluate all terms to a 2D array of { percent, offset } structures
        let evaluatedTerms = terms.map(term => {
            if (term.type === 'literal') {
                return term.val.split(';').map(animStr =>
                    animStr.split(',').map(frameStr => {
                        let p = 0, o = 0;
                        if (frameStr.includes('%')) {
                            let parts = frameStr.split('%');
                            p = (parseFloat(parts[0]) / 100) || 0;
                            o = parseFloat(parts[1]) || 0;
                        } else {
                            o = parseFloat(frameStr) || 0;
                        }
                        return { p, o };
                    })
                );
            } else {
                let varStr = this.parser.variables[term.name];
                let varData = varStr ? this.parsePO(varStr) : [[{ p: 0, o: 0 }]];
                return varData.map(anim =>
                    anim.map(frame => ({
                        p: frame.p * term.coeff,
                        o: frame.o * term.coeff
                    }))
                );
            }
        });

        // Add the structures mathematically across identical frame indices
        let maxAnims = Math.max(...evaluatedTerms.map(t => t.length));
        let result = [];
        for (let a = 0; a < maxAnims; a++) {
            let maxFrames = Math.max(...evaluatedTerms.map(t => {
                let anim = t[Math.min(a, t.length - 1)];
                return anim.length;
            }));
            let animResult = [];
            for (let f = 0; f < maxFrames; f++) {
                let sumP = 0, sumO = 0;
                for (let term of evaluatedTerms) {
                    let anim = term[Math.min(a, term.length - 1)];
                    let frame = anim[Math.min(f, anim.length - 1)];
                    sumP += frame.p;
                    sumO += frame.o;
                }
                animResult.push({ p: sumP, o: sumO });
            }
            result.push(animResult);
        }
        return result;
    }

    parseLiteral(valStr) {
        let p = 0, o = 0;
        if (valStr.includes('%')) {
            let parts = valStr.split('%');
            p = (parseFloat(parts[0]) / 100) || 0;
            o = parseFloat(parts[1]) || 0;
        } else {
            o = parseFloat(valStr) || 0;
        }
        return { p, o };
    }

    evalMathExpr(expr, aIdx, fIdx) {
        let tokens = [];
        // Updated regex to safely catch all decimal formats and % combos
        let regex = /((?:[0-9]*\.[0-9]+|[0-9]+)(?:%(?:[0-9]*\.[0-9]+|[0-9]+))?)|([a-zA-Z_][a-zA-Z0-9_]*)|([\+\-\*\/\^\(\)])/g;
        let m;
        while ((m = regex.exec(expr)) !== null) {
            if (m[1]) tokens.push({ type: 'num', val: m[1] });
            else if (m[2]) tokens.push({ type: 'var', val: m[2] });
            else if (m[3]) tokens.push({ type: 'op', val: m[3] });
        }

        let processed = [];
        for (let i = 0; i < tokens.length; i++) {
            let t = tokens[i];
            processed.push(t);
            if (i < tokens.length - 1) {
                let next = tokens[i + 1];
                let tIsVal = t.type === 'num' || t.type === 'var' || t.val === ')';
                let nextIsVal = next.type === 'num' || next.type === 'var' || next.val === '(';
                if (tIsVal && nextIsVal) {
                    processed.push({ type: 'op', val: '*' });
                }
            }
        }

        let pos = 0;
        const peek = () => processed[pos];
        const consume = () => processed[pos++];

        const parseExpr = () => parseAddSub();

        const parseAddSub = () => {
            let node = parseMulDiv();
            while (peek() && (peek().val === '+' || peek().val === '-')) {
                node = { type: 'binop', op: consume().val, left: node, right: parseMulDiv() };
            }
            return node;
        };

        const parseMulDiv = () => {
            let node = parseExponent();
            while (peek() && (peek().val === '*' || peek().val === '/')) {
                node = { type: 'binop', op: consume().val, left: node, right: parseExponent() };
            }
            return node;
        };

        const parseExponent = () => {
            let node = parseUnary();
            while (peek() && peek().val === '^') {
                node = { type: 'binop', op: consume().val, left: node, right: parseUnary() };
            }
            return node;
        };

        const parseUnary = () => {
            if (peek() && (peek().val === '+' || peek().val === '-')) {
                return { type: 'unary', op: consume().val, expr: parseUnary() };
            }
            return parsePrimary();
        };

        const parsePrimary = () => {
            let t = consume();
            if (!t) return { type: 'num', val: { p: 0, o: 0 } };
            if (t.val === '(') {
                let expr = parseExpr();
                if (peek() && peek().val === ')') consume();
                return expr;
            }
            if (t.type === 'num') return { type: 'num', val: this.parseLiteral(t.val) };
            if (t.type === 'var') return { type: 'var', val: t.val };
            return { type: 'num', val: { p: 0, o: 0 } };
        };

        const evalAST = (node) => {
            if (node.type === 'num') return node.val;
            if (node.type === 'var') {
                let varStr = this.parser.variables[node.val];
                if (!varStr) return { p: 0, o: 0 };

                let animStrs = varStr.split(';');
                let animStr = animStrs[Math.min(aIdx, animStrs.length - 1)] || "";
                let frameStrs = animStr.split(',');
                let frameStr = frameStrs[Math.min(fIdx, frameStrs.length - 1)] || "";

                // Strip stray braces just in case, and route directly back into evalMathExpr
                frameStr = frameStr.replace(/[{}]/g, '');
                return this.evalMathExpr(frameStr, aIdx, fIdx);
            }
            if (node.type === 'unary') {
                let val = evalAST(node.expr);
                if (node.op === '-') return { p: -val.p, o: -val.o };
                return val;
            }
            if (node.type === 'binop') {
                let left = evalAST(node.left);
                let right = evalAST(node.right);
                if (node.op === '+') return { p: left.p + right.p, o: left.o + right.o };
                if (node.op === '-') return { p: left.p - right.p, o: left.o - right.o };
                if (node.op === '*') return { p: left.p * right.o + right.p * left.o, o: left.o * right.o };
                if (node.op === '/') {
                    let denom = right.o !== 0 ? right.o : 1;
                    return { p: left.p / denom, o: left.o / denom };
                }
                if (node.op === '^') return { p: 0, o: Math.pow(left.o, right.o) };
            }
            return { p: 0, o: 0 };
        };

        return evalAST(parseExpr());
    }

    evaluateFrameString(str, aIdx, fIdx) {
        let sumP = 0, sumO = 0;
        let regex = /\{([^}]+)\}|([^\s{}]+)/g;
        let m;
        while ((m = regex.exec(str)) !== null) {
            if (m[1]) {
                let val = this.evalMathExpr(m[1], aIdx, fIdx);
                sumP += val.p;
                sumO += val.o;
            } else if (m[2]) {
                let val = this.parseLiteral(m[2]);
                sumP += val.p;
                sumO += val.o;
            }
        }
        return { p: sumP, o: sumO };
    }

    parseValue(valStr, axis) {
        let animStrs = valStr.split(';');
        return animStrs.map((animStr, aIdx) => {
            let frameStrs = animStr.split(',');
            return frameStrs.map((frameStr, fIdx) => {
                let frameVal = this.evaluateFrameString(frameStr.trim(), aIdx, fIdx);
                if (frameVal.p !== 0) {
                    if (axis === 'x') this.hasRelativeX = true;
                    if (axis === 'y') this.hasRelativeY = true;
                    if (axis === 'none') {
                        this.hasRelativeX = true;
                        this.hasRelativeY = true;
                    }
                }
                return (w, h) => frameVal.p * (axis === 'x' ? w : (axis === 'y' ? h : Math.min(w, h))) + frameVal.o;
            });
        });
    }

    resolveFrames(animations) {
        for (let anim of animations) {
            for (let i = 0; i < anim.length; i++) {
                if (anim[i] === null && i > 0) anim[i] = anim[i - 1];
            }
            for (let i = anim.length - 1; i >= 0; i--) {
                if (anim[i] === null && i < anim.length - 1) anim[i] = anim[i + 1];
            }
        }
    }

    parse() {
        let i = 0;
        let initLevel = 0;
        let currentCommand = null;
        let blockIdCounter = 0;
        let activeBlocks = [0];
        let isFlipping = false; // Tracks if we have crossed the pipe

        let currentPath = {
            commands: [],
            fill: null,
            stroke: { width: this.parseValue("1%", "vmin"), color: null },
            animBehavior: null,
            blockTypes: { 0: 'forward' }
        };

        let flushPath = () => {
            if (currentPath.commands.length > 0) {
                this.paths.push(currentPath);
                currentPath = {
                    commands: [],
                    fill: currentPath.fill,
                    stroke: currentPath.stroke,
                    animBehavior: currentPath.animBehavior,
                    blockTypes: { 0: 'forward' }
                };
            }
        };

        while (i < this.tokens.length) {
            let tok = this.tokens[i];

            if (tok.type === 'command') {
                if (tok.value === '[') {
                    initLevel++;
                    blockIdCounter++;
                    // If we crossed a pipe, new nested brackets are automatically reverse blocks
                    currentPath.blockTypes[blockIdCounter] = isFlipping ? 'reverse' : 'forward';
                    activeBlocks.push(blockIdCounter);
                    i++;
                } else if (tok.value === ']') {
                    initLevel--;
                    activeBlocks.pop();
                    // Reset the flipping state when we return to the root level
                    if (initLevel === 0) isFlipping = false;
                    i++;
                } else if (tok.value === '|') {
                    isFlipping = true;
                    // Wipe the current forward blocks from the active stack and recreate them 
                    // identically as reverse blocks so everything connected from here draws backwards
                    let newActiveBlocks = [0];
                    for (let k = 1; k <= initLevel; k++) {
                        blockIdCounter++;
                        currentPath.blockTypes[blockIdCounter] = 'reverse';
                        newActiveBlocks.push(blockIdCounter);
                    }
                    activeBlocks = newActiveBlocks;
                    i++;
                } else if (tok.value === 'F' || tok.value === 'f') {
                    flushPath();
                    i++;
                    let colors = [];
                    while (i < this.tokens.length && this.tokens[i].type === 'string') {
                        colors.push(this.tokens[i].value);
                        i++;
                    }
                    if (colors.length === 0) {
                        currentPath.fill = null;
                    } else if (colors.length === 1) {
                        currentPath.fill = colors[0];
                    } else {
                        currentPath.fill = { type: tok.value === 'F' ? 'linear' : 'radial', colors };
                    }
                } else if (tok.value === 'B' || tok.value === 'b') {
                    flushPath();
                    i++;
                    let args = [];
                    while (i < this.tokens.length && (this.tokens[i].type === 'value' || this.tokens[i].type === 'string')) {
                        args.push(this.tokens[i]);
                        i++;
                    }
                    if (args.length === 0) {
                        currentPath.stroke = null;
                    } else {
                        currentPath.stroke = {
                            width: args[0].type === 'value' ? this.parseValue(args[0].value, 'vmin') : null,
                            color: args.length > 1 ? args[1].value : null
                        };
                    }
                } else if (tok.value === 'D' || tok.value === 'd') {
                    flushPath();
                    let isInitAttr = tok.value === 'd';
                    i++;
                    let behaviorStr = "";
                    while (i < this.tokens.length && this.tokens[i].type === 'value') {
                        behaviorStr += (behaviorStr ? " " : "") + this.tokens[i].value;
                        i++;
                    }
                    if (isInitAttr) {
                        currentPath.initBehavior = behaviorStr;
                    } else {
                        currentPath.animBehavior = behaviorStr;
                    }
                } else if (tok.value === 'W' || tok.value === 'w') {
                    let isRel = tok.value === 'w';
                    i++;
                    let args = [];
                    while (i < this.tokens.length && this.tokens[i].type === 'value') {
                        args.push(parseFloat(this.tokens[i].value));
                        i++;
                    }
                    if (isRel) {
                        this.sizeRatio = { w: args[0] || 1, h: args[1] || 1 };
                    } else {
                        this.fixedSize = { w: args[0] || 100, h: args[1] || 100 };
                    }
                    // Inside XPath.parse()
                } else if (COMMAND_ARGS[tok.value]) {
                    let isOffset = tok.value === 'O' || tok.value === 'o';
                    let cmdToUse = tok.value;

                    // Only update the persistent flag if it isn't an offset command
                    if (!isOffset) {
                        currentCommand = tok.value;
                    }

                    let axes = COMMAND_ARGS[cmdToUse];

                    if (axes.length === 0) {
                        currentPath.commands.push({ type: 'Z', args: [], initLevel, blockId: activeBlocks[activeBlocks.length - 1] });
                        i++;
                    } else {
                        i++;
                        let args = [];
                        while (i < this.tokens.length && this.tokens[i].type === 'value') {
                            let valStr = this.tokens[i].value;
                            let axis = axes[args.length % axes.length];
                            let mapped = this.parseValue(valStr, axis);
                            this.resolveFrames(mapped);
                            args.push(mapped);
                            i++;
                            if (args.length % axes.length === 0) {
                                currentPath.commands.push({ type: cmdToUse, args: args.slice(), initLevel, blockId: activeBlocks[activeBlocks.length - 1] });
                                args = [];
                                if (cmdToUse === 'M') { currentCommand = 'L'; cmdToUse = 'L'; }
                                if (cmdToUse === 'm') { currentCommand = 'l'; cmdToUse = 'l'; }

                                // Process exactly one set of coordinates for offsets, then break
                                // to return control to the previously active command flag
                                if (isOffset) {
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    currentCommand = tok.value;
                    i++;
                }
            } else if (tok.type === 'value') {
                if (currentCommand && COMMAND_ARGS[currentCommand] && COMMAND_ARGS[currentCommand].length > 0) {
                    let axes = COMMAND_ARGS[currentCommand];
                    let args = [];
                    while (i < this.tokens.length && this.tokens[i].type === 'value') {
                        let valStr = this.tokens[i].value;
                        let axis = axes[args.length % axes.length];
                        let mapped = this.parseValue(valStr, axis);
                        this.resolveFrames(mapped);
                        args.push(mapped);
                        i++;
                        console.log(`Pushed arg ${valStr}, index is now ${i}`);
                        if (args.length % axes.length === 0) {
                            currentPath.commands.push({ type: currentCommand, args: args.slice(), initLevel, blockId: activeBlocks[activeBlocks.length - 1] });
                            args = [];
                            if (currentCommand === 'M') currentCommand = 'L';
                            if (currentCommand === 'm') currentCommand = 'l';
                        }
                    }
                } else {
                    i++;
                }
            } else {
                i++;
            }
        }
        flushPath();
    }

    getMaxInitLevel() {
        let max = 0;
        for (let path of this.paths) {
            for (let cmd of path.commands) {
                if (cmd.initLevel > max) max = cmd.initLevel;
            }
        }
        return max;
    }

    getDString(path, width, height, animIndex, frameIndex, initCoeff) {
        let maxLevel = this.getMaxInitLevel();
        let t = initCoeff * maxLevel;

        let dStrs = [];
        let truePen = { x: 0, y: 0 };
        let trueM = { x: 0, y: 0 };

        let blockRenderStartPoints = { 0: { x: 0, y: 0 } };
        let blockRenderEndPoints = {}; // Computed later for {}

        // Pass 1: Compute absolute fully resolved positions to find endpoints
        let tempPensByCommand = [];
        let currentOffsetX = 0;
        let currentOffsetY = 0;

        for (let i = 0; i < path.commands.length; i++) {
            let cmd = path.commands[i];
            let targetArgs = cmd.args.map(argFuncs => {
                let animGrp = argFuncs[Math.min(animIndex, argFuncs.length - 1)];
                let func = animGrp[Math.min(frameIndex, animGrp.length - 1)];
                return func(width, height);
            });
            let type = cmd.type;
            let upperType = type.toUpperCase();
            let isRel = type === type.toLowerCase();

            if (upperType === 'O') {
                if (isRel) {
                    currentOffsetX += targetArgs[0];
                    currentOffsetY += targetArgs[1];
                } else {
                    currentOffsetX = targetArgs[0];
                    currentOffsetY = targetArgs[1];
                }
                tempPensByCommand.push({ x: truePen.x, y: truePen.y, absArgs: [], isOffset: true });
                continue;
            }

            // Calculate and store absolute arguments for curve points etc.
            let absArgs = [];
            let prevPenX = i > 0 ? tempPensByCommand[i - 1].x : 0;
            let prevPenY = i > 0 ? tempPensByCommand[i - 1].y : 0;
            if (upperType === 'Z') { /* no args */ }
            else if (upperType === 'H') { absArgs[0] = (isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX; }
            else if (upperType === 'V') { absArgs[0] = (isRel ? prevPenY - currentOffsetY + targetArgs[0] : targetArgs[0]) + currentOffsetY; }
            else if (upperType === 'M' || upperType === 'L' || upperType === 'T') {
                absArgs[0] = (isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX;
                absArgs[1] = (isRel ? prevPenY - currentOffsetY + targetArgs[1] : targetArgs[1]) + currentOffsetY;
            } else if (upperType === 'C') {
                absArgs[0] = (isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX;
                absArgs[1] = (isRel ? prevPenY - currentOffsetY + targetArgs[1] : targetArgs[1]) + currentOffsetY;
                absArgs[2] = (isRel ? prevPenX - currentOffsetX + targetArgs[2] : targetArgs[2]) + currentOffsetX;
                absArgs[3] = (isRel ? prevPenY - currentOffsetY + targetArgs[3] : targetArgs[3]) + currentOffsetY;
                // For curve endpoint mapping, we'll store its absolute coordinate below after it updates truePen
            } else if (upperType === 'S' || upperType === 'Q') {
                absArgs[0] = (isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX;
                absArgs[1] = (isRel ? prevPenY - currentOffsetY + targetArgs[1] : targetArgs[1]) + currentOffsetY;
            } else if (upperType === 'A') {
                absArgs[0] = targetArgs[0]; absArgs[1] = targetArgs[1];
                absArgs[2] = targetArgs[2]; absArgs[3] = targetArgs[3]; absArgs[4] = targetArgs[4];
            }

            // Update truePen first
            if (upperType === 'Z') { truePen.x = trueM.x; truePen.y = trueM.y; }
            else if (upperType === 'H') { truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX; }
            else if (upperType === 'V') { truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[0] : targetArgs[0]) + currentOffsetY; }
            else if (upperType === 'M' || upperType === 'L' || upperType === 'T') {
                truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX;
                truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[1] : targetArgs[1]) + currentOffsetY;
                if (upperType === 'M') { trueM.x = truePen.x; trueM.y = truePen.y; }
            } else if (upperType === 'C') {
                truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[4] : targetArgs[4]) + currentOffsetX;
                truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[5] : targetArgs[5]) + currentOffsetY;
            } else if (upperType === 'S' || upperType === 'Q') {
                truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[2] : targetArgs[2]) + currentOffsetX;
                truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[3] : targetArgs[3]) + currentOffsetY;
            } else if (upperType === 'A') {
                truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[5] : targetArgs[5]) + currentOffsetX;
                truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[6] : targetArgs[6]) + currentOffsetY;
            }

            if (upperType === 'C') {
                absArgs[4] = truePen.x; absArgs[5] = truePen.y;
            } else if (upperType === 'S' || upperType === 'Q') {
                absArgs[2] = truePen.x; absArgs[3] = truePen.y;
            } else if (upperType === 'A') {
                absArgs[5] = truePen.x; absArgs[6] = truePen.y;
            }

            tempPensByCommand.push({ x: truePen.x, y: truePen.y, absArgs: absArgs });
        }

        // Find indices boundaries for each block
        let blockIndices = {}; // bId -> { startIdx, endIdx }
        for (let i = 0; i < path.commands.length; i++) {
            let bId = path.commands[i].blockId;
            if (!blockIndices[bId]) blockIndices[bId] = { startIdx: i, endIdx: i };
            blockIndices[bId].endIdx = i;
        }

        let renderedPointsMemo = {}; // Memoization cache for getRenderedPoint

        // Recursive function to get the rendered (scaled) position of a command's endpoint
        let getRenderedPoint = (cmdIdx) => {
            // Base case: point before the first command (origin)
            if (cmdIdx < 0) return { x: 0, y: 0 };
            // Base case: point after the last command (last rendered point)
            if (cmdIdx >= path.commands.length) {
                // If there are no commands, return origin. Otherwise, return the last command's rendered point.
                return path.commands.length > 0 ? getRenderedPoint(path.commands.length - 1) : { x: 0, y: 0 };
            }

            if (renderedPointsMemo[cmdIdx]) return renderedPointsMemo[cmdIdx];

            let cmd = path.commands[cmdIdx];
            let level = cmd.initLevel;
            let bId = cmd.blockId;
            let type = path.blockTypes[bId] || 'forward';

            let cL = 1;
            if (level > 0) {
                cL = Math.max(0, Math.min(1, t - (level - 1)));
            }

            let shrunkPt;
            if (bId === 0) { // Global block, shrinks to origin
                shrunkPt = { x: 0, y: 0 };
            } else if (type === 'forward') {
                let sIdx = blockIndices[bId].startIdx;
                // Shrunk point is the rendered point of the command *before* this block started
                shrunkPt = getRenderedPoint(sIdx - 1);
            } else { // reverse block
                let eIdx = blockIndices[bId].endIdx;
                // Shrunk point is the rendered point of the command *after* this block ends
                shrunkPt = getRenderedPoint(eIdx + 1);
            }

            let truePen = tempPensByCommand[cmdIdx];
            let rx = (1 - cL) * shrunkPt.x + cL * truePen.x;
            let ry = (1 - cL) * shrunkPt.y + cL * truePen.y;

            let result = { x: rx, y: ry, shrunkPt, cL };
            renderedPointsMemo[cmdIdx] = result;
            return result;
        };

        let renderPen = { x: 0, y: 0 };
        let renderM = { x: 0, y: 0 };
        truePen = { x: 0, y: 0 };
        trueM = { x: 0, y: 0 };

        for (let i = 0; i < path.commands.length; i++) {
            let cmd = path.commands[i];
            let upperType = cmd.type.toUpperCase();

            let pt = getRenderedPoint(i);
            let nx = Number(pt.x.toFixed(3));
            let ny = Number(pt.y.toFixed(3));
            let cL = pt.cL;
            let shrunkPt = pt.shrunkPt;

            let absArgs = tempPensByCommand[i].absArgs;

            // Interpolation function for arguments (control points, radii)
            let interpX = (val) => Number(((1 - cL) * shrunkPt.x + cL * val).toFixed(3));
            let interpY = (val) => Number(((1 - cL) * shrunkPt.y + cL * val).toFixed(3));

            let dPart = "";
            let isOffset = tempPensByCommand[i].isOffset;
            if (isOffset) continue;

            if (upperType === 'Z') {
                dPart = 'Z';
                renderPen.x = renderM.x; renderPen.y = renderM.y;
            } else if (upperType === 'M' || upperType === 'L' || upperType === 'T' || upperType === 'H' || upperType === 'V') {
                dPart = (upperType === 'M' ? 'M ' : 'L ') + nx + ' ' + ny;
                if (upperType === 'M') {
                    renderM = { x: nx, y: ny };
                    trueM = { x: truePen.x, y: truePen.y };
                }
                renderPen.x = nx; renderPen.y = ny;
            } else if (upperType === 'C') {
                let x1 = interpX(absArgs[0]), y1 = interpY(absArgs[1]);
                let x2 = interpX(absArgs[2]), y2 = interpY(absArgs[3]);
                dPart = 'C ' + x1 + ' ' + y1 + ', ' + x2 + ' ' + y2 + ', ' + nx + ' ' + ny;
                renderPen.x = nx; renderPen.y = ny;
            } else if (upperType === 'S' || upperType === 'Q') {
                let x1 = interpX(absArgs[0]), y1 = interpY(absArgs[1]);
                dPart = upperType + ' ' + x1 + ' ' + y1 + ', ' + nx + ' ' + ny;
                renderPen.x = nx; renderPen.y = ny;
            } else if (upperType === 'A') {
                let rx = Number((cL * absArgs[0]).toFixed(3));
                let ry = Number((cL * absArgs[1]).toFixed(3));
                let rX0 = rx === 0 ? 0.001 : rx;
                let rY0 = ry === 0 ? 0.001 : ry;
                let rot = absArgs[2], large = absArgs[3], sweep = absArgs[4];
                dPart = 'A ' + rX0 + ' ' + rY0 + ' ' + rot + ' ' + large + ' ' + sweep + ' ' + nx + ' ' + ny;
                renderPen.x = nx; renderPen.y = ny;
            }
            dStrs.push(dPart);
        }
        return dStrs.join(' ');
    }

    parseDBehavior(str) {
        if (!str) return null;
        let p = str.split(' ');
        let timesStr = p[0];
        let splicesStrs = p.slice(1);
        let times = timesStr.split(';').map(animDur => animDur.split(',').map(Number));
        let splices = splicesStrs.map(s => s.split(';').map(ss => ss.split(' ')));
        return { timesStr, splicesStrs, times, splices };
    }

    get(prop) {
        if (prop === 'animations') {
            let numAnims = 0;
            for (let path of this.paths) {
                for (let cmd of path.commands) {
                    for (let arg of cmd.args) {
                        if (arg.length > 1 && arg.length > numAnims) {
                            numAnims = arg.length;
                        } else if (arg.length === 1 && arg[0].length > 1 && numAnims === 0) {
                            numAnims = 1;
                        }
                    }
                }
            }
            return numAnims;
        } else if (prop === 'loading') {
            for (let path of this.paths) {
                if (Object.keys(path.blockTypes).length > 1) return true;
            }
            return false;
        } else if (prop === 'width' || prop === 'height') {
            if (prop === 'width' && this.hasRelativeX) return null;
            if (prop === 'height' && this.hasRelativeY) return null;
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            let numAnims = Math.max(1, this.get('animations'));

            for (let path of this.paths) {
                for (let aIdx = 0; aIdx < numAnims; aIdx++) {
                    let numFrames = 1;
                    for (let cmd of path.commands) {
                        for (let arg of cmd.args) {
                            let animGrp = arg[Math.min(aIdx, arg.length - 1)];
                            if (animGrp.length > numFrames) numFrames = animGrp.length;
                        }
                    }
                    for (let fIdx = 0; fIdx < numFrames; fIdx++) {
                        let truePen = { x: 0, y: 0 };
                        let trueM = { x: 0, y: 0 };

                        let addPt = (x, y) => {
                            if (x < minX) minX = x;
                            if (x > maxX) maxX = x;
                            if (y < minY) minY = y;
                            if (y > maxY) maxY = y;
                        };

                        let currentOffsetX = 0;
                        let currentOffsetY = 0;

                        for (let i = 0; i < path.commands.length; i++) {
                            let cmd = path.commands[i];
                            let targetArgs = cmd.args.map(argFuncs => {
                                let animGrp = argFuncs[Math.min(aIdx, argFuncs.length - 1)];
                                let func = animGrp[Math.min(fIdx, animGrp.length - 1)];
                                return func(0, 0);
                            });
                            let type = cmd.type;
                            let upperType = type.toUpperCase();
                            let isRel = type === type.toLowerCase();

                            if (upperType === 'O') {
                                if (isRel) {
                                    currentOffsetX += targetArgs[0];
                                    currentOffsetY += targetArgs[1];
                                } else {
                                    currentOffsetX = targetArgs[0];
                                    currentOffsetY = targetArgs[1];
                                }
                                continue;
                            }

                            let prevPenX = truePen.x;
                            let prevPenY = truePen.y;

                            if (upperType === 'Z') { truePen.x = trueM.x; truePen.y = trueM.y; }
                            else if (upperType === 'H') { truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX; }
                            else if (upperType === 'V') { truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[0] : targetArgs[0]) + currentOffsetY; }
                            else if (upperType === 'M' || upperType === 'L' || upperType === 'T') {
                                truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX;
                                truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[1] : targetArgs[1]) + currentOffsetY;
                                if (upperType === 'M') { trueM.x = truePen.x; trueM.y = truePen.y; }
                            } else if (upperType === 'C') {
                                addPt((isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX, (isRel ? prevPenY - currentOffsetY + targetArgs[1] : targetArgs[1]) + currentOffsetY);
                                addPt((isRel ? prevPenX - currentOffsetX + targetArgs[2] : targetArgs[2]) + currentOffsetX, (isRel ? prevPenY - currentOffsetY + targetArgs[3] : targetArgs[3]) + currentOffsetY);
                                truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[4] : targetArgs[4]) + currentOffsetX;
                                truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[5] : targetArgs[5]) + currentOffsetY;
                            } else if (upperType === 'S' || upperType === 'Q') {
                                addPt((isRel ? prevPenX - currentOffsetX + targetArgs[0] : targetArgs[0]) + currentOffsetX, (isRel ? prevPenY - currentOffsetY + targetArgs[1] : targetArgs[1]) + currentOffsetY);
                                truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[2] : targetArgs[2]) + currentOffsetX;
                                truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[3] : targetArgs[3]) + currentOffsetY;
                            } else if (upperType === 'A') {
                                truePen.x = (isRel ? prevPenX - currentOffsetX + targetArgs[5] : targetArgs[5]) + currentOffsetX;
                                truePen.y = (isRel ? prevPenY - currentOffsetY + targetArgs[6] : targetArgs[6]) + currentOffsetY;
                            }

                            if (upperType !== 'Z') {
                                addPt(truePen.x, truePen.y);
                            }
                        }
                    }
                }
            }
            if (minX === Infinity) return 0;
            if (prop === 'width') return maxX - minX;
            if (prop === 'height') return maxY - minY;
        }
        return null;
    }

    gets(prop) {
        return this.get(prop);
    }

    code(width, height) {
        let svg = '';
        for (let pIdx = 0; pIdx < this.paths.length; pIdx++) {
            let path = this.paths[pIdx];

            let attrs = [];
            if (path.fill) {
                if (typeof path.fill === 'string') {
                    attrs.push(`fill="${path.fill}"`);
                } else {
                    attrs.push(`fill="${path.fill.colors[0]}"`);
                }
            } else {
                attrs.push('fill="none"');
            }

            if (path.stroke) {
                let sw = path.stroke.width[0][0](width, height);
                attrs.push(`stroke-width="${sw}"`);
                if (path.stroke.color) {
                    attrs.push(`stroke="${path.stroke.color}"`);
                } else {
                    attrs.push(`stroke="#000"`);
                }
            }

            let dInitial = this.getDString(path, width, height, 0, 0, 0);

            svg += `  <path id="xpath_${pIdx}" d="${dInitial}" ${attrs.join(' ')}>\n`;

            let numAnims = this.get('animations');
            numAnims = Math.max(1, numAnims);

            for (let aIdx = 0; aIdx < numAnims; aIdx++) {
                let numFrames = 1;
                for (let cmd of path.commands) {
                    for (let arg of cmd.args) {
                        let animGrp = arg[Math.min(aIdx, arg.length - 1)];
                        if (animGrp.length > numFrames) numFrames = animGrp.length;
                    }
                }

                let values = [];
                for (let fIdx = 0; fIdx < numFrames; fIdx++) {
                    values.push(this.getDString(path, width, height, aIdx, fIdx, 1.0));
                }

                if (values.length > 1 || numAnims > 1) {
                    svg += `    <animate id="xpath_${pIdx}_anim_${aIdx}" attributeName="d" values="${values.join(';')}" begin="indefinite" fill="freeze" />\n`;
                }
            }

            svg += `  </path>\n`;
        }
        return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">\n${svg}</svg>`;
    }
}

if (typeof window !== 'undefined') {
    window.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('script[type="text/xpath"]').forEach(script => {
            let xpath = new XPath(script.textContent);

            let getDim = () => {
                let w = script.clientWidth || script.parentElement.clientWidth || 500;
                let h = script.clientHeight || script.parentElement.clientHeight || 500;
                if (xpath.fixedSize) {
                    w = xpath.fixedSize.w;
                    h = xpath.fixedSize.h;
                } else if (xpath.sizeRatio) {
                    let pw = script.parentElement.clientWidth || 500;
                    let ph = script.parentElement.clientHeight || 500;
                    let vmin = Math.min(pw / xpath.sizeRatio.w, ph / xpath.sizeRatio.h);
                    w = vmin * xpath.sizeRatio.w;
                    h = vmin * xpath.sizeRatio.h;
                }
                return { w, h };
            };

            let dims = getDim();
            let width = dims.w;
            let height = dims.h;

            let svgStr = xpath.code(width, height);

            let temp = document.createElement('div');
            temp.innerHTML = svgStr;
            let svgEl = temp.firstElementChild;

            svgEl.style.cssText = script.style.cssText;
            svgEl.style.display = 'block';
            svgEl.id = script.id ? script.id + '_svg' : 'xpath_svg_' + Math.random().toString(36).substr(2, 9);

            script.parentElement.insertBefore(svgEl, script);

            let currentCoeff = 0;
            let currentAnim = 0;
            let currentFrame = 0;

            // Function to re-render a specific state
            let renderCurrentState = () => {
                let dims = getDim();
                let w = dims.w;
                let h = dims.h;

                // Update SVG container dims
                svgEl.setAttribute('width', w);
                svgEl.setAttribute('height', h);

                // We recreate the SVG paths/animations internally to handle resize
                let newSvgStr = xpath.code(w, h);
                let newTemp = document.createElement('div');
                newTemp.innerHTML = newSvgStr;
                let newSvgEl = newTemp.firstElementChild;

                // We can just swap the innerHTML of svgEl to update all paths & anims
                svgEl.innerHTML = newSvgEl.innerHTML;

                // If animating 'init' state manually (coeff < 1), apply explicit D strings
                for (let pIdx = 0; pIdx < xpath.paths.length; pIdx++) {
                    let pathEl = svgEl.querySelector(`#xpath_${pIdx}`);
                    if (pathEl) {
                        pathEl.setAttribute('d', xpath.getDString(xpath.paths[pIdx], w, h, currentAnim, currentFrame, currentCoeff));
                    }
                }
            };

            // Observe dimension changes
            let resizeObserver = new ResizeObserver(() => {
                renderCurrentState();
            });
            // Observe the parent since script tag might have 0 dimensions depending on display
            resizeObserver.observe(script.parentElement);
            if (script.style.width || script.style.height) {
                resizeObserver.observe(script);
            }

            script.load = function (cf = 1, dr) {
                if (dr === undefined) {
                    dr = 1;
                    if (xpath.paths[0] && xpath.paths[0].initBehavior) {
                        let parsed = xpath.parseDBehavior(xpath.paths[0].initBehavior);
                        if (parsed && parsed.times[0] && parsed.times[0][0]) {
                            dr = parsed.times[0][0];
                        }
                    }
                }

                let startTime = performance.now();
                let startCoeff = currentCoeff;

                function frame(time) {
                    let elapsed = (time - startTime) / 1000;
                    let progress = Math.min(1, elapsed / dr);
                    let coeff = startCoeff + (cf - startCoeff) * progress;
                    currentCoeff = coeff;

                    let dims = getDim();
                    let w = dims.w;
                    let h = dims.h;

                    for (let pIdx = 0; pIdx < xpath.paths.length; pIdx++) {
                        let pathEl = svgEl.querySelector(`#xpath_${pIdx}`);
                        if (pathEl) {
                            pathEl.setAttribute('d', xpath.getDString(xpath.paths[pIdx], w, h, currentAnim, currentFrame, coeff));
                        }
                    }
                    if (progress < 1) requestAnimationFrame(frame);
                }
                requestAnimationFrame(frame);
            };

            script.animate = function (id = 0, dr, rp = 1, bn = false) {
                currentAnim = id;
                for (let pIdx = 0; pIdx < xpath.paths.length; pIdx++) {
                    let animEl = svgEl.querySelector(`#xpath_${pIdx}_anim_${id}`);
                    if (animEl) {
                        let targetDr = dr;
                        if (targetDr === undefined) {
                            targetDr = 1;
                            if (xpath.paths[pIdx].animBehavior) {
                                let parsed = xpath.parseDBehavior(xpath.paths[pIdx].animBehavior);
                                if (parsed && parsed.times[id] && parsed.times[id][0]) {
                                    targetDr = parsed.times[id][0];
                                }
                            }
                        }

                        animEl.setAttribute('dur', targetDr + 's');

                        if (bn) {
                            let origValues = animEl.getAttribute('data-orig-values');
                            if (!origValues) {
                                origValues = animEl.getAttribute('values');
                                animEl.setAttribute('data-orig-values', origValues);
                            }
                            let vArr = origValues.split(';');
                            let rev = vArr.slice(0, -1).reverse();
                            animEl.setAttribute('values', vArr.concat(rev).join(';'));
                        }

                        if (rp === 0) animEl.setAttribute('repeatCount', 'indefinite');
                        else if (rp > 0) animEl.setAttribute('repeatCount', rp);

                        animEl.beginElement();
                    }
                }
            };

            script.get = (prop) => xpath.get(prop);
            script.gets = (prop) => xpath.gets(prop);
        });
    });
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { XPathParser, XPath };
}
