(function () {
    const xpathRegistry = new WeakMap();
    const activeXpaths = new Set();

    // -- Parsers --
    const parseDur = (str) => (str ? str.split(';').map(s => parseFloat(s.trim()) * 1000 || 1000) : [1000]);
    const parseKeyTimes = (str) => (str ? str.split(';').map(s => s.split(',').map(n => parseFloat(n.trim()))) : []);

    function parseFrame(f, isX) {
        const match = f.match(/^(-?\d+(?:\.\d+)?)%(?:([+-]?\d+(?:\.\d+)?))?/);
        if (match) {
            const percent = parseFloat(match[1]) / 100;
            const offset = match[2] ? parseFloat(match[2]) : 0;
            return (dimX, dimY) => percent * (isX ? dimX : dimY) + offset;
        } else {
            const val = parseFloat(f);
            return () => val;
        }
    }

    function parseStates(tokenStr, isX) {
        return tokenStr.split(':').map(stateStr => {
            let clean = stateStr.trim();
            const autoStart = !clean.startsWith(';');
            if (!autoStart) clean = clean.substring(1);
            
            const loop = !clean.endsWith(';');
            if (!loop) clean = clean.substring(0, clean.length - 1);
            
            const frames = clean.split(',').map(f => parseFrame(f.trim(), isX));
            return { loop, autoStart, frames };
        });
    }

    function parseD(dStr) {
        if (!dStr) return [];
        dStr = dStr.replace(/\([^)]*\)/g, '').trim();
        
        const regex = /([MmLlHhVvCcSsQqTtAaZz])|([;]?[-+]?\d[\d.%,;:\-+]*)/g;
        const tokens = [];
        let match;
        let currentCmd = 'M';
        let argIndex = 0;

        while ((match = regex.exec(dStr)) !== null) {
            if (match[1]) {
                currentCmd = match[1];
                argIndex = 0;
                tokens.push({ type: 'cmd', val: currentCmd });
            } else if (match[2]) {
                const cmdUpper = currentCmd.toUpperCase();
                let isX = true;
                
                if (cmdUpper === 'A') {
                    const mapA = [true, false, true, true, true, true, false];
                    isX = mapA[argIndex % 7];
                } else if (cmdUpper === 'V') isX = false;
                else if (cmdUpper === 'H') isX = true;
                else isX = (argIndex % 2) === 0; 

                tokens.push({ type: 'val', states: parseStates(match[2], isX), currentVal: null, startVal: null });
                argIndex++;
            }
        }
        return tokens;
    }

    // -- Initialization & Control --
    function initXPath(el) {
        if (xpathRegistry.has(el)) return;

        el.style.display = 'none';
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const svg = el.ownerSVGElement;
        
        for (let attr of el.attributes) {
            if (!['id', 'd', 'dur', 'keyTimes', 'style'].includes(attr.name)) {
                path.setAttribute(attr.name, attr.value);
            }
        }
        el.parentNode.insertBefore(path, el.nextSibling);

        const xpData = {
            el, path, svg,
            currentState: 0,
            triggerTime: performance.now(),
            customDur: null,
            started: false,
            lastD: '',
            tokens: parseD(el.getAttribute('d')),
            durs: parseDur(el.getAttribute('dur')),
            keyTimes: parseKeyTimes(el.getAttribute('keyTimes'))
        };

        // Attach custom animate function
        el.animate = function(stateIndex, customDurSeconds) {
            // CRITICAL FIX: Cache the exact current values so we can transition seamlessly to the next state
            for (let t of xpData.tokens) {
                if (t.type === 'val') {
                    t.startVal = t.currentVal !== null ? t.currentVal : null;
                }
            }
            xpData.currentState = stateIndex;
            xpData.triggerTime = performance.now();
            xpData.customDur = customDurSeconds !== undefined ? customDurSeconds * 1000 : null;
            xpData.started = true;
        };

        if (!svg.__resizeObserver) {
            svg.__resizeObserver = new ResizeObserver(() => { svg.__forceUpdate = true; });
            svg.__resizeObserver.observe(svg);
        }

        xpathRegistry.set(el, xpData);
        activeXpaths.add(xpData);
    }

    const observer = new MutationObserver(mutations => {
        mutations.forEach(m => {
            if (m.type === 'childList') {
                m.addedNodes.forEach(node => {
                    if (node.tagName && node.tagName.toLowerCase() === 'xpath') initXPath(node);
                    else if (node.querySelectorAll) node.querySelectorAll('xpath').forEach(initXPath);
                });
            }
            if (m.type === 'attributes' && m.target.tagName.toLowerCase() === 'xpath') {
                const xp = xpathRegistry.get(m.target);
                if (!xp) return;
                const attr = m.attributeName;
                if (attr === 'd') xp.tokens = parseD(m.target.getAttribute('d'));
                else if (attr === 'dur') xp.durs = parseDur(m.target.getAttribute('dur'));
                else if (attr === 'keyTimes') xp.keyTimes = parseKeyTimes(m.target.getAttribute('keyTimes'));
                else if (!['id', 'style'].includes(attr)) xp.path.setAttribute(attr, m.target.getAttribute(attr));
            }
        });
    });

    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['d', 'dur', 'keyTimes', 'stroke', 'fill'] });
    document.querySelectorAll('xpath').forEach(initXPath);

    // -- Master Animation Loop --
    function animationLoop(time) {
        for (let xp of activeXpaths) {
            const svgRect = xp.svg.getBoundingClientRect();
            const dimX = svgRect.width;
            const dimY = svgRect.height;
            let dStr = '';
            
            for (let t of xp.tokens) {
                if (t.type === 'cmd') {
                    dStr += t.val + ' ';
                    continue;
                }

                const sIdx = Math.min(xp.currentState, t.states.length - 1);
                const state = t.states[sIdx];
                const frames = state.frames;
                const dur = xp.customDur || xp.durs[sIdx] || xp.durs[0] || 1000;
                let valToRender = 0;

                // Single-frame state (e.g. 100%) - Transition from the last cached position
                if (frames.length === 1) {
                    const targetVal = frames[0](dimX, dimY);
                    
                    if (xp.started && t.startVal !== null) {
                        let progress = Math.min(1, Math.max(0, (time - xp.triggerTime) / dur));
                        valToRender = t.startVal + (targetVal - t.startVal) * progress;
                    } else {
                        valToRender = targetVal;
                    }
                } 
                // Multi-frame animation (e.g. 0,100%,50) - Run specific internal sequence
                else {
                    let progress = (time - xp.triggerTime) / dur;

                    if (progress >= 1) progress = state.loop ? (progress % 1) : 1;
                    if (!xp.started && !state.autoStart) progress = 0;

                    let kt = xp.keyTimes[sIdx];
                    if (!kt || kt.length !== frames.length) {
                        kt = frames.map((_, i) => i / (frames.length - 1)); 
                    }

                    let idx = 0;
                    for (let i = 0; i < kt.length - 1; i++) {
                        if (progress >= kt[i] && progress <= kt[i + 1]) {
                            idx = i;
                            break;
                        }
                    }

                    let segProgress = 0;
                    if (kt[idx + 1] !== kt[idx]) {
                        segProgress = (progress - kt[idx]) / (kt[idx + 1] - kt[idx]);
                    }

                    const val1 = frames[idx](dimX, dimY);
                    const val2 = frames[idx + 1](dimX, dimY);
                    valToRender = val1 + (val2 - val1) * segProgress;
                }

                // Save current value to allow seamless transitions on the next trigger
                t.currentVal = valToRender; 
                dStr += valToRender + ' ';
            }

            dStr = dStr.trim();
            if (dStr !== xp.lastD || xp.svg.__forceUpdate) {
                xp.path.setAttribute('d', dStr);
                xp.lastD = dStr;
            }
            xp.svg.__forceUpdate = false;
        }
        requestAnimationFrame(animationLoop);
    }

    requestAnimationFrame(animationLoop);
})();