(() => {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const mark = document.querySelector('#hero-mark');
  const settled = mark.firstElementChild;
  const svgNamespace = 'http://www.w3.org/2000/svg';
  const assembly = document.createElementNS(svgNamespace, 'g');
  assembly.setAttribute('fill', 'currentColor');

  function fraction(seed) {
    const value = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
    return value - Math.floor(value);
  }

  const rings = [
    { radius: 42, count: 16, offset: 0 },
    { radius: 56, count: 20, offset: 9 },
    { radius: 70, count: 26, offset: 4 },
  ];

  const dots = rings.flatMap(({ radius, count, offset }, ringIndex) =>
    Array.from({ length: count }, (_, index) => {
      const seed = ringIndex * 100 + index;
      const angle = (index / count) * Math.PI * 2 + (offset * Math.PI) / 180 - Math.PI / 2;
      const delay = ringIndex * 0.45 + (index / count) * 0.9 + fraction(seed + 99) * 0.15;
      const spin = (fraction(seed + 7) - 0.5) * 2;
      const startRadius = 200 + fraction(seed + 50) * 180;
      const element = document.createElementNS(svgNamespace, 'circle');
      assembly.append(element);
      return { element, angle, delay, spin, radius, startRadius };
    })
  );

  const flight = 2.2;
  const duration = 4.2;
  settled.setAttribute('visibility', 'hidden');
  mark.append(assembly);

  let start;
  function assemble(now) {
    start ??= now;
    const elapsed = (now - start) / 1000;
    if (elapsed >= duration) {
      assembly.remove();
      settled.removeAttribute('visibility');
      return;
    }

    assembly.setAttribute('transform', `rotate(${-12 * (1 - elapsed / duration) ** 3})`);
    for (const { element, angle, delay, spin, radius, startRadius } of dots) {
      const progress = Math.min(1, Math.max(0, (elapsed - 0.3 - delay) / flight));
      const remaining = (1 - progress) ** 4;
      const currentAngle = angle + spin * remaining;
      const currentRadius = radius + (startRadius - radius) * remaining;
      element.setAttribute('cx', Math.cos(currentAngle) * currentRadius);
      element.setAttribute('cy', Math.sin(currentAngle) * currentRadius);
      element.setAttribute('r', 5.5 * (1 - 0.4 * remaining));
      element.setAttribute('opacity', Math.min(1, progress * 4));
    }
    requestAnimationFrame(assemble);
  }

  requestAnimationFrame(assemble);
})();
