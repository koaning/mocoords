// js/parcoords.js
function render({ model, el }) {
  const container = document.createElement("div");
  const chartContainer = document.createElement("div");
  chartContainer.className = "chart-container";
  const canvas = document.createElement("canvas");
  canvas.id = "parallelCoordinatesCanvas";
  chartContainer.appendChild(canvas);
  container.appendChild(chartContainer);
  const infoText = document.createElement("p");
  infoText.className = "info-text";
  infoText.textContent = "Click/drag on axis to create/resize brush. Click/drag existing brush rectangle to move it.";
  container.appendChild(infoText);
  const buttonContainer = document.createElement("div");
  buttonContainer.className = "button-container";
  const resetButton = document.createElement("button");
  resetButton.id = "resetButton";
  resetButton.className = "button button-reset";
  resetButton.textContent = "Reset";
  const subsetButton = document.createElement("button");
  subsetButton.id = "subsetButton";
  subsetButton.className = "button button-subset";
  subsetButton.disabled = true;
  subsetButton.textContent = "Subset to Selection";
  const excludeButton = document.createElement("button");
  excludeButton.id = "excludeButton";
  excludeButton.className = "button button-exclude";
  excludeButton.disabled = true;
  excludeButton.textContent = "Exclude Selection";
  buttonContainer.appendChild(resetButton);
  buttonContainer.appendChild(subsetButton);
  buttonContainer.appendChild(excludeButton);
  container.appendChild(buttonContainer);
  el.appendChild(container);
  const data = model.get("data");
  function generateData(count) {
    const data2 = [];
    console.log(`Generating ${count} data points...`);
    const startTime = performance.now();
    for (let i = 0; i < count; i++) {
      const dimA = Math.random() * 100;
      const dimB = 50 + Math.random() * 100;
      const dimC = Math.random() > 0.5 ? Math.random() * 50 : 50 + Math.random() * 50;
      const dimD = dimA * 0.5 + Math.random() * 50;
      const dimE = 200 - dimB * 0.8 + Math.random() * 40;
      let group = dimA < 33 ? "groupA" : dimA < 66 ? "groupB" : "groupC";
      data2.push({ dimA, dimB, dimC, dimD, dimE, group });
    }
    const endTime = performance.now();
    console.log(`Data generation took ${(endTime - startTime).toFixed(1)} ms`);
    return data2;
  }
  console.log(data);
  const originalData = Object.freeze(model.get("data").map((d) => ({ ...d, color: d.color || "default" })));
  const DATA_COUNT = originalData.length;
  const ctx = canvas.getContext("2d");
  const bgCanvas = document.createElement("canvas");
  const bgCtx = bgCanvas.getContext("2d");
  const fgCanvas = document.createElement("canvas");
  const fgCtx = fgCanvas.getContext("2d");
  const margin = { top: 40, right: 30, bottom: 30, left: 30 };
  const dimensions = Object.keys(originalData[0] || {}).filter((key) => key !== "color");
  const groupColors = {};
  const baseColors = [
    "rgba(59, 130, 246, 1)",
    // blue
    "rgba(239, 68, 68, 1)",
    // red
    "rgba(16, 185, 129, 1)",
    // green
    "rgba(217, 119, 6, 1)",
    // orange
    "rgba(139, 92, 246, 1)",
    // purple
    "rgba(236, 72, 153, 1)"
    // pink
  ];
  const uniqueGroups = [...new Set(originalData.map((d) => d.color))];
  uniqueGroups.forEach((group, index) => {
    groupColors[group] = baseColors[index % baseColors.length];
  });
  console.log(uniqueGroups);
  console.log(groupColors);
  const defaultLineAlpha = Math.max(5e-3, Math.min(0.05, 30 / DATA_COUNT));
  const defaultLineColor = `rgba(156, 163, 175, ${defaultLineAlpha})`;
  const activeLineAlpha = Math.max(0.05, Math.min(0.7, 500 / DATA_COUNT));
  const axisColor = "#6b7280";
  const labelColor = "#1f2937";
  const brushColor = "rgba(107, 114, 128, 0.3)";
  const interactionAreaColor = "rgba(100, 100, 255, 0.07)";
  const axisLabelFont = "14px 'Helvetica Neue', Arial, sans-serif";
  const axisTickFont = "10px 'Helvetica Neue', Arial, sans-serif";
  const axisWidthThreshold = 15;
  let width, height, plotWidth, plotHeight;
  let xScales = {};
  let yScales = {};
  let brushes = {};
  let isBrushing = false;
  let isDraggingBrush = false;
  let brushAxis = null;
  let brushStartY = null;
  let dragStartOffsetY = 0;
  let draggedBrushInitialExtent = null;
  let currentData = [];
  let backgroundNeedsRedraw = true;
  let previousBrushState = false;
  function linearScale(domainMin, domainMax, rangeMin, rangeMax) {
    return function(value) {
      const clampedValue = Math.max(domainMin, Math.min(domainMax, value));
      if (domainMax === domainMin)
        return (rangeMin + rangeMax) / 2;
      return rangeMin + (clampedValue - domainMin) * (rangeMax - rangeMin) / (domainMax - domainMin);
    };
  }
  function drawBackgroundLayer(isActiveBrush) {
    console.log(`Drawing background layer (brush active: ${isActiveBrush})...`);
    const startTime = performance.now();
    bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
    bgCtx.lineWidth = 0.5;
    currentData.forEach((d) => {
      let strokeStyle;
      if (isActiveBrush) {
        strokeStyle = defaultLineColor;
      } else {
        const color = groupColors[d.color] || defaultLineColor;
        strokeStyle = color.replace(/[\d\.]+\)$/g, `${activeLineAlpha})`);
      }
      bgCtx.strokeStyle = strokeStyle;
      bgCtx.beginPath();
      dimensions.forEach((dim, i) => {
        const x = xScales[dim];
        const y = yScales[dim].scaleFunc(d[dim]);
        if (i === 0)
          bgCtx.moveTo(x, y);
        else
          bgCtx.lineTo(x, y);
      });
      bgCtx.stroke();
    });
    backgroundNeedsRedraw = false;
    const endTime = performance.now();
    console.log(`Background layer took ${(endTime - startTime).toFixed(1)} ms`);
  }
  function draw() {
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    const isBrushActive = Object.keys(brushes).length > 0;
    if (backgroundNeedsRedraw || isBrushActive !== previousBrushState) {
      drawBackgroundLayer(isBrushActive);
      previousBrushState = isBrushActive;
    }
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    if (bgCanvas.width > 0 && bgCanvas.height > 0) {
      ctx.drawImage(bgCanvas, 0, 0, width, height);
    } else {
      console.warn("Background canvas has invalid dimensions, skipping draw");
    }
    if (isBrushActive) {
      if (fgCanvas.width > 0 && fgCanvas.height > 0) {
        fgCtx.clearRect(0, 0, fgCanvas.width, fgCanvas.height);
        const highlightedData = getHighlightedData();
        fgCtx.lineWidth = 1;
        highlightedData.forEach((d) => {
          const color = groupColors[d.color] || defaultLineColor;
          fgCtx.strokeStyle = color.replace(/[\d\.]+\)$/g, `${activeLineAlpha})`);
          fgCtx.beginPath();
          dimensions.forEach((dim, i) => {
            const x = xScales[dim];
            const y = yScales[dim].scaleFunc(d[dim]);
            if (i === 0)
              fgCtx.moveTo(x, y);
            else
              fgCtx.lineTo(x, y);
          });
          fgCtx.stroke();
        });
        ctx.drawImage(fgCanvas, 0, 0, width, height);
      } else {
        console.warn("Foreground canvas has invalid dimensions, skipping draw");
      }
    }
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = axisColor;
    ctx.fillStyle = labelColor;
    ctx.font = axisLabelFont;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    dimensions.forEach((dim) => {
      const x = xScales[dim];
      const yRangeMin = margin.top + plotHeight;
      const yRangeMax = margin.top;
      ctx.beginPath();
      ctx.moveTo(x, yRangeMax);
      ctx.lineTo(x, yRangeMin);
      ctx.stroke();
      ctx.fillText(dim, x, margin.top - 15);
      ctx.font = axisTickFont;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(yScales[dim].min.toFixed(1), x - 4, yRangeMin);
      ctx.fillText(yScales[dim].max.toFixed(1), x - 4, yRangeMax);
      ctx.font = axisLabelFont;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
    });
    ctx.fillStyle = interactionAreaColor;
    dimensions.forEach((dim) => {
      const x = xScales[dim];
      ctx.fillRect(x - axisWidthThreshold / 2, margin.top, axisWidthThreshold, plotHeight);
    });
    ctx.fillStyle = brushColor;
    Object.entries(brushes).forEach(([dim, extent]) => {
      if (extent) {
        const x = xScales[dim];
        const y1 = extent[0];
        const y2 = extent[1];
        ctx.fillRect(x - axisWidthThreshold / 2, Math.min(y1, y2), axisWidthThreshold, Math.abs(y2 - y1));
      }
    });
    subsetButton.disabled = !isBrushActive;
    excludeButton.disabled = !isBrushActive;
  }
  function getHighlightedData() {
    const activeBrushes = Object.entries(brushes).filter(([_, extent]) => extent);
    if (activeBrushes.length === 0)
      return [];
    return currentData.filter((d) => {
      return activeBrushes.every(([dim, extent]) => {
        const scaleInfo = yScales[dim];
        const val = d[dim];
        const yPos = scaleInfo.scaleFunc(val);
        const brushMinY = Math.min(extent[0], extent[1]);
        const brushMaxY = Math.max(extent[0], extent[1]);
        return yPos >= brushMinY && yPos <= brushMaxY;
      });
    });
  }
  function getMousePos(canvas2, evt) {
    const rect = canvas2.getBoundingClientRect();
    return {
      x: evt.clientX - rect.left,
      y: evt.clientY - rect.top
    };
  }
  function getAxisUnderCursor(mouseX_logical) {
    let foundDim = null;
    for (const dim of dimensions) {
      const axisX_logical = xScales[dim];
      const halfThreshold = axisWidthThreshold / 2;
      if (mouseX_logical >= axisX_logical - halfThreshold && mouseX_logical <= axisX_logical + halfThreshold) {
        foundDim = dim;
        break;
      }
    }
    return foundDim;
  }
  function handleMouseDown(event) {
    const pos = getMousePos(canvas, event);
    let clickedOnExistingBrush = false;
    for (const dim in brushes) {
      if (brushes.hasOwnProperty(dim)) {
        const extent = brushes[dim];
        if (!extent)
          continue;
        const axisX_logical = xScales[dim];
        const halfThreshold = axisWidthThreshold / 2;
        const minXRange = axisX_logical - halfThreshold;
        const maxXRange = axisX_logical + halfThreshold;
        const brushMinY = Math.min(extent[0], extent[1]);
        const brushMaxY = Math.max(extent[0], extent[1]);
        if (pos.x >= minXRange && pos.x <= maxXRange && pos.y >= brushMinY && pos.y <= brushMaxY) {
          clickedOnExistingBrush = true;
          isDraggingBrush = true;
          isBrushing = false;
          brushAxis = dim;
          draggedBrushInitialExtent = [...extent];
          dragStartOffsetY = pos.y - brushMinY;
          canvas.classList.add("dragging");
          break;
        }
      }
    }
    if (!clickedOnExistingBrush) {
      const axis = getAxisUnderCursor(pos.x);
      if (axis) {
        const wasAlreadyBrushing = Object.keys(brushes).length > 0;
        isBrushing = true;
        isDraggingBrush = false;
        brushAxis = axis;
        brushStartY = Math.max(margin.top, Math.min(margin.top + plotHeight, pos.y));
        brushes[brushAxis] = [brushStartY, brushStartY];
        canvas.classList.add("brushing");
        if (!wasAlreadyBrushing) {
          previousBrushState = false;
        }
        draw();
      } else {
        if (Object.keys(brushes).length > 0) {
          brushes = {};
          isBrushing = false;
          isDraggingBrush = false;
          draw();
        }
      }
    }
  }
  function getHighlightedData() {
    if (Object.keys(brushes).length === 0) {
      return currentData;
    }
    return currentData.filter((d) => {
      return Object.entries(brushes).every(([dim, extent]) => {
        const value = d[dim];
        const yScale = yScales[dim].scaleFunc;
        const valueY = yScale(value);
        const brushMinY = Math.min(extent[0], extent[1]);
        const brushMaxY = Math.max(extent[0], extent[1]);
        return valueY >= brushMinY && valueY <= brushMaxY;
      });
    });
  }
  function handleMouseMove(event) {
    if (isDraggingBrush) {
      requestAnimationFrame(() => {
        if (!isDraggingBrush)
          return;
        const pos = getMousePos(canvas, event);
        const brushHeight = Math.abs(draggedBrushInitialExtent[0] - draggedBrushInitialExtent[1]);
        let newTopY = pos.y - dragStartOffsetY;
        let newBottomY = newTopY + brushHeight;
        if (newTopY < margin.top) {
          newTopY = margin.top;
          newBottomY = newTopY + brushHeight;
        }
        if (newBottomY > margin.top + plotHeight) {
          newBottomY = margin.top + plotHeight;
          newTopY = newBottomY - brushHeight;
          newTopY = Math.max(margin.top, newTopY);
        }
        brushes[brushAxis] = [newTopY, newBottomY];
        draw();
      });
    } else if (isBrushing) {
      requestAnimationFrame(() => {
        if (!isBrushing)
          return;
        const pos = getMousePos(canvas, event);
        const currentY = Math.max(margin.top, Math.min(margin.top + plotHeight, pos.y));
        if (brushes[brushAxis] && brushes[brushAxis][1] !== currentY) {
          brushes[brushAxis][1] = currentY;
          draw();
        }
      });
    }
  }
  function handleMouseUp(event) {
    const wasBrushing = isBrushing;
    if (isBrushing || isDraggingBrush) {
      if (wasBrushing && brushes[brushAxis] && Math.abs(brushes[brushAxis][0] - brushes[brushAxis][1]) < 1) {
        delete brushes[brushAxis];
      }
      isBrushing = false;
      isDraggingBrush = false;
      const anyBrushRemaining = Object.keys(brushes).length > 0;
      if (!anyBrushRemaining && previousBrushState) {
        previousBrushState = true;
      } else if (anyBrushRemaining && !previousBrushState) {
        previousBrushState = false;
      }
      brushAxis = null;
      brushStartY = null;
      dragStartOffsetY = 0;
      draggedBrushInitialExtent = null;
      canvas.classList.remove("brushing", "dragging");
      draw();
      model.set("selection", getHighlightedData());
      model.save_changes();
    }
  }
  function handleReset() {
    currentData = [...originalData];
    brushes = {};
    isBrushing = false;
    isDraggingBrush = false;
    backgroundNeedsRedraw = true;
    canvas.classList.remove("brushing", "dragging");
    draw();
  }
  function handleSubset() {
    const isBrushActiveInitially = Object.keys(brushes).length > 0;
    if (!isBrushActiveInitially)
      return;
    const highlighted = getHighlightedData();
    if (highlighted.length > 0 && highlighted.length < currentData.length) {
      currentData = highlighted;
      brushes = {};
      isBrushing = false;
      isDraggingBrush = false;
      backgroundNeedsRedraw = true;
      canvas.classList.remove("brushing", "dragging");
      draw();
    } else if (isBrushActiveInitially) {
      brushes = {};
      isBrushing = false;
      isDraggingBrush = false;
      canvas.classList.remove("brushing", "dragging");
      draw();
    }
  }
  function handleExclude() {
    const isBrushActiveInitially = Object.keys(brushes).length > 0;
    if (!isBrushActiveInitially)
      return;
    const highlighted = getHighlightedData();
    if (highlighted.length > 0 && highlighted.length < currentData.length) {
      const highlightedSet = new Set(highlighted);
      currentData = currentData.filter((d) => !highlightedSet.has(d));
      brushes = {};
      isBrushing = false;
      isDraggingBrush = false;
      backgroundNeedsRedraw = true;
      canvas.classList.remove("brushing", "dragging");
      draw();
    } else if (isBrushActiveInitially) {
      if (highlighted.length === currentData.length) {
        currentData = [];
      }
      brushes = {};
      isBrushing = false;
      isDraggingBrush = false;
      backgroundNeedsRedraw = true;
      canvas.classList.remove("brushing", "dragging");
      draw();
    }
  }
  function setup() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    if (width <= 0 || height <= 0) {
      console.log("Canvas dimensions are zero, waiting for resize...");
      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const newWidth = entry.contentRect.width;
          const newHeight = entry.contentRect.height;
          if (newWidth > 0 && newHeight > 0) {
            console.log(`Canvas resized to ${newWidth}x${newHeight}`);
            resizeObserver.disconnect();
            setup();
            return;
          }
        }
      });
      resizeObserver.observe(canvas);
      return;
    }
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    bgCanvas.width = canvas.width;
    bgCanvas.height = canvas.height;
    bgCtx.resetTransform();
    bgCtx.scale(dpr, dpr);
    fgCanvas.width = canvas.width;
    fgCanvas.height = canvas.height;
    fgCtx.resetTransform();
    fgCtx.scale(dpr, dpr);
    plotWidth = width - margin.left - margin.right;
    plotHeight = height - margin.top - margin.bottom;
    xScales = {};
    dimensions.forEach((dim, i) => {
      xScales[dim] = margin.left + i * (plotWidth / (dimensions.length - 1));
    });
    yScales = {};
    dimensions.forEach((dim) => {
      const domain = originalData.reduce((acc, d) => [
        Math.min(acc[0], d[dim]),
        Math.max(acc[1], d[dim])
      ], [Infinity, -Infinity]);
      if (domain[0] === domain[1]) {
        domain[0] -= 0.5;
        domain[1] += 0.5;
      }
      yScales[dim] = {
        min: domain[0],
        max: domain[1],
        scaleFunc: linearScale(domain[0], domain[1], margin.top + plotHeight, margin.top)
      };
    });
    if (currentData.length === 0) {
      currentData = [...originalData];
    }
    brushes = {};
    isBrushing = false;
    isDraggingBrush = false;
    previousBrushState = false;
    brushAxis = null;
    brushStartY = null;
    dragStartOffsetY = 0;
    draggedBrushInitialExtent = null;
    canvas.classList.remove("brushing", "dragging");
    canvas.style.cursor = "default";
    backgroundNeedsRedraw = true;
    draw();
  }
  canvas.addEventListener("mousedown", handleMouseDown);
  canvas.addEventListener("mousemove", (event) => {
    if (isBrushing || isDraggingBrush) {
      event.preventDefault();
      handleMouseMove(event);
    }
  });
  canvas.addEventListener("mouseup", handleMouseUp);
  canvas.addEventListener("mouseleave", handleMouseUp);
  resetButton.addEventListener("click", handleReset);
  subsetButton.addEventListener("click", handleSubset);
  excludeButton.addEventListener("click", handleExclude);
  let resizeTimeout;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      backgroundNeedsRedraw = true;
      setup();
    }, 150);
  });
  setup();
}
var parcoords_default = { render };
export {
  parcoords_default as default
};
