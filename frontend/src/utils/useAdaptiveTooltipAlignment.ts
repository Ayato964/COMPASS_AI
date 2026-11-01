/**
 * Viewport & Container-Aware Adaptive Tooltip Alignment + WCAG 2.1 SC 1.4.13 Hover Bridge & Escape Dismissal
 * (`useAdaptiveTooltipAlignment.ts`).
 * - Prevents horizontal & vertical clipping (`0px` overflow) of `button[data-tooltip]::after` across
 *   all flex-wrap permutations (`320px`..`4K`) and scrollable sidebars/modals (`aside`, dialog cards).
 * - Keeps `button[data-tooltip]::after` strictly `pointer-events: none` so neighboring controls (`Solo`/`Mute`,
 *   `<select>`, wrapped toolbar rows, `<canvas>` ruler scrub & `AI TARGET` handles) are NEVER occluded (`[R8-BUG-01]`),
 *   while maintaining WCAG 2.1 SC 1.4.13 Hoverable persistence via a lightweight `[data-tooltip-bridge="true"]`
 *   geometric corridor.
 * - Allows `Escape` to dismiss an active hovered/bridged/`:focus-visible` tooltip without closing underlying
 *   modals on the 1st press, while never swallowing `<canvas>` drag cancellation or `<input>` draft revert (`[R8-BUG-03]`).
 */

import { useEffect } from 'react';

interface TooltipCorridor {
  readonly btnLeft: number;
  readonly btnRight: number;
  readonly btnTop: number;
  readonly btnBottom: number;
  readonly tipLeft: number;
  readonly tipRight: number;
  readonly isTop: boolean;
}

const DISCRETE_CONTROL_SELECTOR = [
  'button:not(:disabled)',
  'select:not(:disabled)',
  'input:not(:disabled)',
  'textarea:not(:disabled)',
  'label',
  'a[href]',
  '[role="button"]',
  '[role="tab"]',
  '[role="checkbox"]',
  '[role="radio"]',
  '[contenteditable="true"]',
].join(', ');

let activeSubscriberCount = 0;
let hoveredTooltipBtn: HTMLButtonElement | null = null;
let hoveredDisabledMarkerBtn: HTMLButtonElement | null = null;
let keyboardFocusTooltipBtn: HTMLButtonElement | null = null;
let bridgedTooltipBtn: HTMLButtonElement | null = null;
let bridgedCorridor: TooltipCorridor | null = null;
let bridgeMoveListenerAttached = false;
let activeTooltipStripEl: HTMLElement | null = null;
const dismissedTooltipButtons = new Set<HTMLButtonElement>();

function isJsdomEnvironment(): boolean {
  return typeof navigator !== 'undefined' && /jsdom/i.test(navigator.userAgent ?? '');
}

function isPointerDragActive(): boolean {
  if (typeof document === 'undefined') return false;
  return document.querySelector('[data-pointer-dragging="true"]') !== null;
}

function isTextEntryElement(el: Element | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  if (el.isContentEditable || el.getAttribute('contenteditable') === 'true') {
    return true;
  }
  if (el.closest('textarea, [contenteditable="true"]')) {
    return true;
  }
  const input = el.closest('input');
  if (input instanceof HTMLInputElement) {
    const t = (input.type || 'text').toLowerCase();
    return (
      t !== 'checkbox' &&
      t !== 'radio' &&
      t !== 'button' &&
      t !== 'submit' &&
      t !== 'reset' &&
      t !== 'range' &&
      t !== 'file'
    );
  }
  return false;
}

function isTooltipSuppressedAtViewport(btn: HTMLButtonElement): boolean {
  if (typeof window === 'undefined') return false;
  if (
    btn.getAttribute('data-tooltip-sm-hidden') === 'true' &&
    (window.innerWidth || document.documentElement.clientWidth || 0) >= 640
  ) {
    return true;
  }
  return false;
}

function canButtonDisplayTooltip(btn: HTMLButtonElement): boolean {
  if (!btn.isConnected) return false;
  if (btn.disabled && btn.getAttribute('data-tooltip-when-disabled') !== 'true') {
    return false;
  }
  const tipText = (btn.getAttribute('data-tooltip') || '').trim();
  if (!tipText) return false;
  if (isTooltipSuppressedAtViewport(btn)) return false;
  if (btn.closest('[hidden], [inert], [aria-hidden="true"]') !== null) {
    return false;
  }
  if (typeof document !== 'undefined') {
    const modal = document.querySelector('[role="dialog"][aria-modal="true"]');
    if (modal && !modal.contains(btn)) {
      return false;
    }
  }
  if (!isJsdomEnvironment() && btn.getClientRects().length === 0) {
    return false;
  }
  return true;
}

function estimateTooltipWidthPx(tipText: string, docWidth: number, inAside = false): number {
  let estPx = 22;
  for (let i = 0; i < tipText.length; i += 1) {
    estPx += tipText.charCodeAt(i) > 0x7f ? 12.5 : 7.5;
  }
  const maxCap = inAside
    ? Math.min(224, Math.max(48, docWidth - 32))
    : Math.min(340, Math.max(48, docWidth - 16));
  return Math.min(maxCap, Math.ceil(estPx));
}

function isOtherDiscreteControl(target: EventTarget | null, originBtn: HTMLButtonElement): boolean {
  if (!(target instanceof Element)) return false;
  const hit = target.closest(DISCRETE_CONTROL_SELECTOR);
  return hit !== null && hit !== originBtn;
}

function findHoveredDisabledTooltipButton(
  clientX: number,
  clientY: number,
  excludeBtn: HTMLButtonElement | null,
  eventTarget?: EventTarget | null,
): HTMLButtonElement | null {
  if (typeof document === 'undefined') return null;
  const candidates = document.querySelectorAll(
    'button[data-tooltip][data-tooltip-when-disabled="true"]:disabled',
  );
  if (candidates.length === 0) return null;
  const hasCoords =
    Number.isFinite(clientX) &&
    Number.isFinite(clientY) &&
    (clientX !== 0 || clientY !== 0);
  let hitContextEl: Element | null = eventTarget instanceof Element ? eventTarget : null;
  if (
    hasCoords &&
    !isJsdomEnvironment() &&
    typeof document.elementFromPoint === 'function' &&
    (hitContextEl === null ||
      hitContextEl === document.documentElement ||
      hitContextEl === document.body)
  ) {
    hitContextEl = document.elementFromPoint(clientX, clientY);
  }
  const checkAncestry =
    hitContextEl !== null &&
    hitContextEl !== document.documentElement &&
    hitContextEl !== document.body;
  for (let i = 0; i < candidates.length; i += 1) {
    const el = candidates[i];
    if (!(el instanceof HTMLButtonElement) || el === excludeBtn) continue;
    if (checkAncestry && hitContextEl && !hitContextEl.contains(el) && !el.contains(hitContextEl)) {
      continue;
    }
    if (hasCoords) {
      const r = el.getBoundingClientRect();
      if (
        r.width > 0 &&
        r.height > 0 &&
        clientX >= r.left &&
        clientX <= r.right &&
        clientY >= r.top &&
        clientY <= r.bottom &&
        canButtonDisplayTooltip(el)
      ) {
        return el;
      }
      continue;
    }
    try {
      if (el.matches(':hover') && canButtonDisplayTooltip(el)) {
        return el;
      }
    } catch {
      // Ignore selector engine error in synthetic DOM environments
    }
  }
  return null;
}

function syncTooltipVisualDomMarkers(): void {
  const nextDisabledHovered =
    hoveredTooltipBtn !== null &&
    hoveredTooltipBtn.disabled &&
    canButtonDisplayTooltip(hoveredTooltipBtn) &&
    hoveredTooltipBtn.getAttribute('data-tooltip-dismissed') !== 'true'
      ? hoveredTooltipBtn
      : null;
  if (hoveredDisabledMarkerBtn !== nextDisabledHovered) {
    hoveredDisabledMarkerBtn?.removeAttribute('data-tooltip-hovered');
    hoveredDisabledMarkerBtn = nextDisabledHovered;
    hoveredDisabledMarkerBtn?.setAttribute('data-tooltip-hovered', 'true');
  }

  let activeBtn: HTMLButtonElement | null = null;
  if (
    bridgedTooltipBtn !== null &&
    canButtonDisplayTooltip(bridgedTooltipBtn) &&
    bridgedTooltipBtn.getAttribute('data-tooltip-dismissed') !== 'true'
  ) {
    activeBtn = bridgedTooltipBtn;
  } else if (
    hoveredTooltipBtn !== null &&
    canButtonDisplayTooltip(hoveredTooltipBtn) &&
    hoveredTooltipBtn.getAttribute('data-tooltip-dismissed') !== 'true'
  ) {
    activeBtn = hoveredTooltipBtn;
  } else if (
    keyboardFocusTooltipBtn !== null &&
    isFocusedButtonShowingTooltip(keyboardFocusTooltipBtn)
  ) {
    activeBtn = keyboardFocusTooltipBtn;
  }
  const rawStrip = activeBtn ? activeBtn.closest('[data-tooltip-strip]') : null;
  const nextStrip = rawStrip instanceof HTMLElement ? rawStrip : null;
  if (activeTooltipStripEl !== nextStrip) {
    activeTooltipStripEl?.removeAttribute('data-tooltip-strip-active');
    activeTooltipStripEl = nextStrip;
    activeTooltipStripEl?.setAttribute('data-tooltip-strip-active', 'true');
  }
}

function computeTooltipCorridor(btn: HTMLButtonElement): TooltipCorridor | null {
  if (typeof window === 'undefined' || typeof document === 'undefined') return null;
  const tipText = (btn.getAttribute('data-tooltip') || '').trim();
  if (!tipText) return null;
  const btnRect = btn.getBoundingClientRect();
  if (btnRect.width <= 0 || btnRect.height <= 0) return null;

  const docWidth = document.documentElement.clientWidth || window.innerWidth || 1024;
  const estWidth = estimateTooltipWidthPx(tipText, docWidth, Boolean(btn.closest('aside')));
  const align = btn.getAttribute('data-tooltip-align');
  let tipLeft: number;
  let tipRight: number;
  if (align === 'start') {
    tipLeft = btnRect.left;
    tipRight = btnRect.left + estWidth;
  } else if (align === 'end') {
    tipRight = btnRect.right;
    tipLeft = btnRect.right - estWidth;
  } else {
    const cx = btnRect.left + btnRect.width * 0.5;
    tipLeft = cx - estWidth * 0.5;
    tipRight = cx + estWidth * 0.5;
  }

  return {
    btnLeft: btnRect.left,
    btnRight: btnRect.right,
    btnTop: btnRect.top,
    btnBottom: btnRect.bottom,
    tipLeft,
    tipRight,
    isTop: btn.getAttribute('data-tooltip-pos') === 'top',
  };
}

function isPointInsideCorridor(x: number, y: number, c: TooltipCorridor): boolean {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  const unionLeft = Math.min(c.btnLeft, c.tipLeft) - 2;
  const unionRight = Math.max(c.btnRight, c.tipRight) + 2;
  if (c.isTop) {
    const inButtonAndGap = y >= c.btnTop - 8 && y <= c.btnBottom + 2 && x >= unionLeft && x <= unionRight;
    const inTipBox =
      y >= c.btnTop - 34 && y <= c.btnTop - 4 && x >= c.tipLeft - 4 && x <= c.tipRight + 4;
    return inButtonAndGap || inTipBox;
  }
  const inButtonAndGap = y >= c.btnTop - 2 && y <= c.btnBottom + 8 && x >= unionLeft && x <= unionRight;
  const inTipBox =
    y >= c.btnBottom + 4 && y <= c.btnBottom + 34 && x >= c.tipLeft - 4 && x <= c.tipRight + 4;
  return inButtonAndGap || inTipBox;
}

function detachBridgeMoveListener(): void {
  if (bridgeMoveListenerAttached && typeof document !== 'undefined') {
    document.removeEventListener('pointermove', handleBridgePointerMove);
    bridgeMoveListenerAttached = false;
  }
}

function clearTooltipBridge(): void {
  if (bridgedTooltipBtn) {
    bridgedTooltipBtn.removeAttribute('data-tooltip-bridge');
    bridgedTooltipBtn = null;
  }
  bridgedCorridor = null;
  detachBridgeMoveListener();
  syncTooltipVisualDomMarkers();
}

function handleBridgePointerMove(e: PointerEvent): void {
  if (!bridgedTooltipBtn || !bridgedCorridor) {
    clearTooltipBridge();
    return;
  }
  if (
    !bridgedTooltipBtn.isConnected ||
    (e.buttons ?? 0) !== 0 ||
    isPointerDragActive() ||
    !canButtonDisplayTooltip(bridgedTooltipBtn) ||
    bridgedTooltipBtn.getAttribute('data-tooltip-dismissed') === 'true'
  ) {
    clearTooltipBridge();
    return;
  }
  if (e.target instanceof Node && bridgedTooltipBtn.contains(e.target)) {
    hoveredTooltipBtn = bridgedTooltipBtn;
    clearTooltipBridge();
    return;
  }
  if (
    bridgedTooltipBtn.disabled &&
    findHoveredDisabledTooltipButton(e.clientX, e.clientY, null, e.target) === bridgedTooltipBtn
  ) {
    hoveredTooltipBtn = bridgedTooltipBtn;
    clearTooltipBridge();
    return;
  }
  if (isOtherDiscreteControl(e.target, bridgedTooltipBtn)) {
    clearTooltipBridge();
    return;
  }
  const disabledHit = findHoveredDisabledTooltipButton(
    e.clientX,
    e.clientY,
    bridgedTooltipBtn,
    e.target,
  );
  if (disabledHit !== null) {
    clearTooltipBridge();
    dismissCompetingFocusTooltips(disabledHit);
    dismissCompetingHoverTooltips(disabledHit);
    clearStaleDismissedTooltips(disabledHit);
    if (hoveredTooltipBtn !== disabledHit) {
      disabledHit.removeAttribute('data-tooltip-dismissed');
      dismissedTooltipButtons.delete(disabledHit);
    }
    hoveredTooltipBtn = disabledHit;
    alignButtonTooltipToViewport(disabledHit);
    syncTooltipVisualDomMarkers();
    return;
  }
  if (!isPointInsideCorridor(e.clientX, e.clientY, bridgedCorridor)) {
    clearTooltipBridge();
  }
}

function activateTooltipBridge(btn: HTMLButtonElement, e: PointerEvent): void {
  const corridor = computeTooltipCorridor(btn);
  if (!corridor) return;
  const hasRealCoords =
    Number.isFinite(e.clientX) &&
    Number.isFinite(e.clientY) &&
    (e.clientX !== 0 || e.clientY !== 0);
  if (hasRealCoords && !isPointInsideCorridor(e.clientX, e.clientY, corridor)) {
    return;
  }
  if (bridgedTooltipBtn && bridgedTooltipBtn !== btn) {
    bridgedTooltipBtn.removeAttribute('data-tooltip-bridge');
  }
  bridgedTooltipBtn = btn;
  bridgedCorridor = corridor;
  btn.setAttribute('data-tooltip-bridge', 'true');
  if (!bridgeMoveListenerAttached && typeof document !== 'undefined') {
    document.addEventListener('pointermove', handleBridgePointerMove, { passive: true });
    bridgeMoveListenerAttached = true;
  }
  syncTooltipVisualDomMarkers();
}

export function alignButtonTooltipToViewport(target: EventTarget | null): void {
  if (typeof window === 'undefined' || typeof document === 'undefined') return;
  if (!(target instanceof Element)) return;
  const btn = target.closest('button[data-tooltip]');
  if (!(btn instanceof HTMLButtonElement)) return;
  if (btn.disabled && btn.getAttribute('data-tooltip-when-disabled') !== 'true') return;
  const tipText = (btn.getAttribute('data-tooltip') || '').trim();
  if (!tipText) return;

  const btnRect = btn.getBoundingClientRect();
  if (btnRect.width <= 0 || btnRect.height <= 0) return;

  const docWidth = document.documentElement.clientWidth || window.innerWidth || 1024;
  const docHeight = document.documentElement.clientHeight || window.innerHeight || 768;
  let boundLeft = 6;
  let boundRight = Math.max(boundLeft + 48, docWidth - 6);
  let boundTop = 4;
  let boundBottom = Math.max(boundTop + 48, docHeight - 4);

  const clipHost = btn.closest('aside, [data-testid="midi-import-modal-card"]');
  if (clipHost instanceof HTMLElement) {
    const hostRect = clipHost.getBoundingClientRect();
    if (hostRect.width > 0) {
      boundLeft = Math.max(boundLeft, hostRect.left + 6);
      boundRight = Math.min(boundRight, hostRect.right - 6);
    }
    if (hostRect.height > 0) {
      boundTop = Math.max(boundTop, hostRect.top + 4);
      boundBottom = Math.min(boundBottom, hostRect.bottom - 4);
      const stickyHdr = clipHost.querySelector('[data-tooltip-sticky-header="true"]');
      if (stickyHdr instanceof HTMLElement && !stickyHdr.contains(btn)) {
        const hdrRect = stickyHdr.getBoundingClientRect();
        if (hdrRect.height > 0) {
          boundTop = Math.max(boundTop, hdrRect.bottom + 4);
        }
      }
    }
  }

  const estWidth = estimateTooltipWidthPx(tipText, docWidth, Boolean(btn.closest('aside')));

  const centerX = btnRect.left + btnRect.width * 0.5;
  const centeredLeft = centerX - estWidth * 0.5;
  const centeredRight = centerX + estWidth * 0.5;

  const calcOverflow = (left: number, right: number): number =>
    Math.max(0, boundLeft - left) + Math.max(0, right - boundRight);

  const centerOverflow = calcOverflow(centeredLeft, centeredRight);
  const startOverflow = calcOverflow(btnRect.left, btnRect.left + estWidth);
  const endOverflow = calcOverflow(btnRect.right - estWidth, btnRect.right);

  if (centerOverflow <= startOverflow && centerOverflow <= endOverflow) {
    btn.removeAttribute('data-tooltip-align');
  } else if (startOverflow <= endOverflow) {
    btn.setAttribute('data-tooltip-align', 'start');
  } else {
    btn.setAttribute('data-tooltip-align', 'end');
  }

  // Adaptive vertical flip (`data-tooltip-pos="top"` <-> default bottom) against viewport & scroll host
  const estHeight = 32;
  const roomAbove = btnRect.top - boundTop;
  const roomBelow = boundBottom - btnRect.bottom;
  const preferTop = btn.getAttribute('data-tooltip-prefer-pos') === 'top';
  if (preferTop) {
    if (roomAbove >= estHeight || roomAbove >= roomBelow) {
      btn.setAttribute('data-tooltip-pos', 'top');
    } else {
      btn.removeAttribute('data-tooltip-pos');
    }
  } else if (roomBelow < estHeight && roomAbove > roomBelow) {
    btn.setAttribute('data-tooltip-pos', 'top');
  } else {
    btn.removeAttribute('data-tooltip-pos');
  }
}

function clearStaleDismissedTooltips(
  exceptBtn: HTMLButtonElement | null,
  forceAll = false,
): void {
  if (dismissedTooltipButtons.size === 0) return;
  const activeEl = typeof document !== 'undefined' ? document.activeElement : null;
  for (const dBtn of Array.from(dismissedTooltipButtons)) {
    if (!dBtn.isConnected) {
      dismissedTooltipButtons.delete(dBtn);
      continue;
    }
    if (dBtn !== exceptBtn && (forceAll || dBtn !== activeEl)) {
      dBtn.removeAttribute('data-tooltip-dismissed');
      dismissedTooltipButtons.delete(dBtn);
    }
  }
}

function dismissCompetingFocusTooltips(exceptBtn: HTMLButtonElement | null): void {
  if (keyboardFocusTooltipBtn && keyboardFocusTooltipBtn !== exceptBtn) {
    dismissButtonTooltip(keyboardFocusTooltipBtn);
    keyboardFocusTooltipBtn = null;
  }
  const activeEl = typeof document !== 'undefined' ? document.activeElement : null;
  if (
    activeEl instanceof HTMLButtonElement &&
    activeEl !== exceptBtn &&
    activeEl.hasAttribute('data-tooltip') &&
    isFocusedButtonShowingTooltip(activeEl)
  ) {
    dismissButtonTooltip(activeEl);
  }
}

function dismissCompetingHoverTooltips(exceptBtn: HTMLButtonElement): void {
  if (hoveredTooltipBtn && hoveredTooltipBtn !== exceptBtn) {
    dismissButtonTooltip(hoveredTooltipBtn);
    hoveredTooltipBtn = null;
  }
  if (typeof document !== 'undefined') {
    try {
      const cssHovered = document.querySelectorAll('button[data-tooltip]:hover');
      for (const el of Array.from(cssHovered)) {
        if (el instanceof HTMLButtonElement && el !== exceptBtn) {
          dismissButtonTooltip(el);
        }
      }
    } catch {
      // Ignore selector engine error in synthetic DOM environments
    }
  }
}

function pruneInvalidTooltipState(): void {
  if (bridgedTooltipBtn && !canButtonDisplayTooltip(bridgedTooltipBtn)) {
    clearTooltipBridge();
  }
  if (hoveredTooltipBtn && !canButtonDisplayTooltip(hoveredTooltipBtn)) {
    hoveredTooltipBtn = null;
  }
  if (keyboardFocusTooltipBtn && !canButtonDisplayTooltip(keyboardFocusTooltipBtn)) {
    keyboardFocusTooltipBtn = null;
  }
  if (dismissedTooltipButtons.size > 0) {
    for (const btn of Array.from(dismissedTooltipButtons)) {
      if (!btn.isConnected) {
        dismissedTooltipButtons.delete(btn);
      }
    }
  }
  syncTooltipVisualDomMarkers();
}

function exitDisabledTooltipHover(
  exiting: HTMLButtonElement,
  e: PointerEvent,
  nextDisabledHit: HTMLButtonElement | null,
): void {
  if (hoveredTooltipBtn === exiting) {
    hoveredTooltipBtn = null;
  }
  if (
    e.pointerType !== 'touch' &&
    (e.buttons ?? 0) === 0 &&
    !isPointerDragActive() &&
    nextDisabledHit === null &&
    canButtonDisplayTooltip(exiting) &&
    exiting.getAttribute('data-tooltip-dismissed') !== 'true' &&
    !isOtherDiscreteControl(e.target, exiting)
  ) {
    activateTooltipBridge(exiting, e);
  }
  if (
    bridgedTooltipBtn !== exiting &&
    exiting !== (typeof document !== 'undefined' ? document.activeElement : null)
  ) {
    exiting.removeAttribute('data-tooltip-dismissed');
    dismissedTooltipButtons.delete(exiting);
  }
}

function handleDocumentPointerOver(e: PointerEvent): void {
  if ((e.buttons ?? 0) !== 0 || isPointerDragActive()) {
    hoveredTooltipBtn = null;
    clearTooltipBridge();
    syncTooltipVisualDomMarkers();
    return;
  }
  const target = e.target;
  if (target instanceof Element) {
    const btn = target.closest('button[data-tooltip]');
    if (btn instanceof HTMLButtonElement) {
      if (bridgedTooltipBtn) {
        clearTooltipBridge();
      }
      if (
        hoveredTooltipBtn?.disabled &&
        hoveredTooltipBtn !== btn &&
        hoveredTooltipBtn !== (typeof document !== 'undefined' ? document.activeElement : null)
      ) {
        hoveredTooltipBtn.removeAttribute('data-tooltip-dismissed');
        dismissedTooltipButtons.delete(hoveredTooltipBtn);
      }
      dismissCompetingFocusTooltips(btn);
      clearStaleDismissedTooltips(btn);
      if (hoveredTooltipBtn !== btn) {
        btn.removeAttribute('data-tooltip-dismissed');
        dismissedTooltipButtons.delete(btn);
      }
      hoveredTooltipBtn = btn;
      alignButtonTooltipToViewport(btn);
      syncTooltipVisualDomMarkers();
      return;
    }
    const disabledBtn = findHoveredDisabledTooltipButton(e.clientX, e.clientY, null, target);
    if (disabledBtn !== null) {
      if (bridgedTooltipBtn) {
        clearTooltipBridge();
      }
      if (
        hoveredTooltipBtn?.disabled &&
        hoveredTooltipBtn !== disabledBtn &&
        hoveredTooltipBtn !== (typeof document !== 'undefined' ? document.activeElement : null)
      ) {
        hoveredTooltipBtn.removeAttribute('data-tooltip-dismissed');
        dismissedTooltipButtons.delete(hoveredTooltipBtn);
      }
      dismissCompetingFocusTooltips(disabledBtn);
      dismissCompetingHoverTooltips(disabledBtn);
      clearStaleDismissedTooltips(disabledBtn);
      if (hoveredTooltipBtn !== disabledBtn) {
        disabledBtn.removeAttribute('data-tooltip-dismissed');
        dismissedTooltipButtons.delete(disabledBtn);
      }
      hoveredTooltipBtn = disabledBtn;
      alignButtonTooltipToViewport(disabledBtn);
      syncTooltipVisualDomMarkers();
      return;
    }
    if (bridgedTooltipBtn && isOtherDiscreteControl(target, bridgedTooltipBtn)) {
      clearTooltipBridge();
    }
  }
  if (hoveredTooltipBtn?.disabled) {
    exitDisabledTooltipHover(hoveredTooltipBtn, e, null);
  } else {
    hoveredTooltipBtn = null;
  }
  if (!bridgedTooltipBtn) {
    clearStaleDismissedTooltips(null);
  }
  syncTooltipVisualDomMarkers();
}

function handleDocumentPointerMove(e: PointerEvent): void {
  if (e.pointerType === 'touch') return;
  if ((e.buttons ?? 0) !== 0 || isPointerDragActive()) {
    if (hoveredTooltipBtn?.disabled || bridgedTooltipBtn) {
      hoveredTooltipBtn = null;
      clearTooltipBridge();
      syncTooltipVisualDomMarkers();
    }
    return;
  }
  const trackingDisabledHover = Boolean(hoveredTooltipBtn && hoveredTooltipBtn.disabled);
  const target = e.target;
  if (target instanceof Element) {
    const directEnabledBtn = target.closest('button[data-tooltip]:not(:disabled)');
    if (directEnabledBtn instanceof HTMLButtonElement) {
      if (trackingDisabledHover) {
        const prevDisabled = hoveredTooltipBtn;
        hoveredTooltipBtn = directEnabledBtn;
        if (
          prevDisabled &&
          prevDisabled !== (typeof document !== 'undefined' ? document.activeElement : null)
        ) {
          prevDisabled.removeAttribute('data-tooltip-dismissed');
          dismissedTooltipButtons.delete(prevDisabled);
        }
        syncTooltipVisualDomMarkers();
      }
      return;
    }
  }

  const disabledHit = findHoveredDisabledTooltipButton(e.clientX, e.clientY, null, target);
  if (!trackingDisabledHover && disabledHit === null) {
    return;
  }
  if (trackingDisabledHover && hoveredTooltipBtn && hoveredTooltipBtn !== disabledHit) {
    exitDisabledTooltipHover(hoveredTooltipBtn, e, disabledHit);
  }

  if (disabledHit !== null) {
    if (bridgedTooltipBtn === disabledHit) {
      hoveredTooltipBtn = disabledHit;
      clearTooltipBridge();
      return;
    }
    if (hoveredTooltipBtn !== disabledHit) {
      if (bridgedTooltipBtn) {
        clearTooltipBridge();
      }
      dismissCompetingFocusTooltips(disabledHit);
      dismissCompetingHoverTooltips(disabledHit);
      clearStaleDismissedTooltips(disabledHit);
      disabledHit.removeAttribute('data-tooltip-dismissed');
      dismissedTooltipButtons.delete(disabledHit);
      hoveredTooltipBtn = disabledHit;
      alignButtonTooltipToViewport(disabledHit);
    }
  }

  syncTooltipVisualDomMarkers();
}

function handleDocumentPointerOut(e: PointerEvent): void {
  const target = e.target;
  if (!(target instanceof Element)) return;
  const btn = target.closest('button[data-tooltip]');
  if (!(btn instanceof HTMLButtonElement)) return;
  const rel = e.relatedTarget;
  if (rel instanceof Node && btn.contains(rel)) return;

  if (hoveredTooltipBtn === btn) {
    hoveredTooltipBtn = null;
  }

  if (
    e.pointerType !== 'touch' &&
    (e.buttons ?? 0) === 0 &&
    !isPointerDragActive() &&
    canButtonDisplayTooltip(btn) &&
    btn.getAttribute('data-tooltip-dismissed') !== 'true' &&
    rel instanceof Element &&
    !isOtherDiscreteControl(rel, btn) &&
    findHoveredDisabledTooltipButton(e.clientX, e.clientY, btn, rel) === null
  ) {
    activateTooltipBridge(btn, e);
    return;
  }

  if (bridgedTooltipBtn === btn) {
    clearTooltipBridge();
  }
  if (btn !== (typeof document !== 'undefined' ? document.activeElement : null)) {
    btn.removeAttribute('data-tooltip-dismissed');
    dismissedTooltipButtons.delete(btn);
  }
  syncTooltipVisualDomMarkers();
}

function handleDocumentPointerDown(e: PointerEvent): void {
  hoveredTooltipBtn = null;
  clearTooltipBridge();
  if (keyboardFocusTooltipBtn) {
    const target = e.target;
    if (!(target instanceof Node) || !keyboardFocusTooltipBtn.contains(target)) {
      keyboardFocusTooltipBtn.setAttribute('data-tooltip-dismissed', 'true');
      dismissedTooltipButtons.add(keyboardFocusTooltipBtn);
    }
    keyboardFocusTooltipBtn = null;
  }
  syncTooltipVisualDomMarkers();
}

function handleDocumentFocusIn(e: FocusEvent): void {
  pruneInvalidTooltipState();
  const target = e.target;
  if (!(target instanceof Element)) return;
  const btn = target.closest('button[data-tooltip]');
  if (btn instanceof HTMLButtonElement) {
    btn.removeAttribute('data-tooltip-dismissed');
    dismissedTooltipButtons.delete(btn);
    alignButtonTooltipToViewport(btn);
    if (isFocusedButtonShowingTooltip(btn)) {
      keyboardFocusTooltipBtn = btn;
      if (bridgedTooltipBtn && bridgedTooltipBtn !== btn) {
        clearTooltipBridge();
      }
      dismissCompetingHoverTooltips(btn);
    } else {
      keyboardFocusTooltipBtn = null;
    }
  } else {
    keyboardFocusTooltipBtn = null;
  }
  syncTooltipVisualDomMarkers();
}

function handleDocumentFocusOut(e: FocusEvent): void {
  const target = e.target;
  if (!(target instanceof Element)) return;
  const btn = target.closest('button[data-tooltip]');
  if (btn instanceof HTMLButtonElement) {
    btn.removeAttribute('data-tooltip-dismissed');
    dismissedTooltipButtons.delete(btn);
    if (keyboardFocusTooltipBtn === btn) {
      keyboardFocusTooltipBtn = null;
    }
  }
  syncTooltipVisualDomMarkers();
}

function handleDocumentPointerLeave(): void {
  hoveredTooltipBtn = null;
  clearTooltipBridge();
  clearStaleDismissedTooltips(null);
  syncTooltipVisualDomMarkers();
}

function handleViewportMutation(): void {
  if (hoveredTooltipBtn?.disabled) {
    hoveredTooltipBtn = null;
  }
  if (bridgedTooltipBtn) {
    clearTooltipBridge();
  }
  syncTooltipVisualDomMarkers();
}

function handleWindowBlur(): void {
  hoveredTooltipBtn = null;
  clearTooltipBridge();
  syncTooltipVisualDomMarkers();
}

function handleVisibilityChange(): void {
  if (typeof document !== 'undefined' && document.hidden) {
    hoveredTooltipBtn = null;
    clearTooltipBridge();
    syncTooltipVisualDomMarkers();
  }
}

function isFocusedButtonShowingTooltip(btn: HTMLButtonElement): boolean {
  if (!canButtonDisplayTooltip(btn)) return false;
  if (btn.getAttribute('data-tooltip-dismissed') === 'true') return false;
  try {
    return btn.matches(':focus-visible');
  } catch {
    return false;
  }
}

function dismissButtonTooltip(btn: HTMLButtonElement): boolean {
  if (!canButtonDisplayTooltip(btn)) return false;
  if (btn.getAttribute('data-tooltip-dismissed') === 'true') return false;
  btn.setAttribute('data-tooltip-dismissed', 'true');
  dismissedTooltipButtons.add(btn);
  return true;
}

function handleWindowKeyUpCapture(): void {
  queueMicrotask(pruneInvalidTooltipState);
}

function handleWindowClickBubble(): void {
  if (
    hoveredTooltipBtn !== null ||
    bridgedTooltipBtn !== null ||
    keyboardFocusTooltipBtn !== null ||
    activeTooltipStripEl !== null
  ) {
    queueMicrotask(pruneInvalidTooltipState);
  }
}

function handleWindowEscapeCapture(e: KeyboardEvent): void {
  if (e.isComposing || e.keyCode === 229) return;
  if (e.key !== 'Escape') {
    queueMicrotask(pruneInvalidTooltipState);
    return;
  }
  let dismissedAny = false;
  if (bridgedTooltipBtn) {
    const bridged = bridgedTooltipBtn;
    clearTooltipBridge();
    if (dismissButtonTooltip(bridged)) {
      dismissedAny = true;
    }
  }
  if (hoveredTooltipBtn && dismissButtonTooltip(hoveredTooltipBtn)) {
    dismissedAny = true;
  }
  const activeEl = typeof document !== 'undefined' ? document.activeElement : null;
  if (typeof document !== 'undefined') {
    try {
      const cssHovered = document.querySelectorAll('button[data-tooltip]:hover');
      for (const el of Array.from(cssHovered)) {
        if (el instanceof HTMLButtonElement && dismissButtonTooltip(el)) {
          dismissedAny = true;
        }
      }
    } catch {
      // Ignore selector engine error in synthetic DOM environments
    }
    if (
      activeEl instanceof HTMLButtonElement &&
      activeEl.hasAttribute('data-tooltip') &&
      isFocusedButtonShowingTooltip(activeEl)
    ) {
      activeEl.setAttribute('data-tooltip-dismissed', 'true');
      dismissedTooltipButtons.add(activeEl);
      if (keyboardFocusTooltipBtn === activeEl) {
        keyboardFocusTooltipBtn = null;
      }
      dismissedAny = true;
    }
  }
  syncTooltipVisualDomMarkers();
  if (dismissedAny && !isPointerDragActive() && !isTextEntryElement(activeEl)) {
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
  }
}

export function useAdaptiveTooltipAlignment(): void {
  useEffect(() => {
    if (typeof document === 'undefined' || typeof window === 'undefined') return undefined;
    if (activeSubscriberCount === 0) {
      document.addEventListener('pointerover', handleDocumentPointerOver, { passive: true });
      document.addEventListener('pointermove', handleDocumentPointerMove, { passive: true });
      document.addEventListener('pointerout', handleDocumentPointerOut, { passive: true });
      document.addEventListener('pointerdown', handleDocumentPointerDown, { passive: true });
      document.addEventListener('focusin', handleDocumentFocusIn, { passive: true });
      document.addEventListener('focusout', handleDocumentFocusOut, { passive: true });
      document.documentElement.addEventListener('pointerleave', handleDocumentPointerLeave, {
        passive: true,
      });
      window.addEventListener('scroll', handleViewportMutation, { capture: true, passive: true });
      window.addEventListener('resize', handleViewportMutation, { passive: true });
      window.addEventListener('blur', handleWindowBlur);
      document.addEventListener('visibilitychange', handleVisibilityChange);
      window.addEventListener('keydown', handleWindowEscapeCapture, true);
      window.addEventListener('keyup', handleWindowKeyUpCapture, true);
      window.addEventListener('click', handleWindowClickBubble, { passive: true });
    }
    activeSubscriberCount += 1;
    return () => {
      activeSubscriberCount = Math.max(0, activeSubscriberCount - 1);
      if (activeSubscriberCount === 0) {
        hoveredTooltipBtn = null;
        keyboardFocusTooltipBtn = null;
        clearTooltipBridge();
        clearStaleDismissedTooltips(null, true);
        dismissedTooltipButtons.clear();
        syncTooltipVisualDomMarkers();
        document.removeEventListener('pointerover', handleDocumentPointerOver);
        document.removeEventListener('pointermove', handleDocumentPointerMove);
        document.removeEventListener('pointerout', handleDocumentPointerOut);
        document.removeEventListener('pointerdown', handleDocumentPointerDown);
        document.removeEventListener('focusin', handleDocumentFocusIn);
        document.removeEventListener('focusout', handleDocumentFocusOut);
        document.documentElement.removeEventListener('pointerleave', handleDocumentPointerLeave);
        window.removeEventListener('scroll', handleViewportMutation, true);
        window.removeEventListener('resize', handleViewportMutation);
        window.removeEventListener('blur', handleWindowBlur);
        document.removeEventListener('visibilitychange', handleVisibilityChange);
        window.removeEventListener('keydown', handleWindowEscapeCapture, true);
        window.removeEventListener('keyup', handleWindowKeyUpCapture, true);
        window.removeEventListener('click', handleWindowClickBubble);
      }
    };
  }, []);
}
