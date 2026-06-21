// Worksite — light progressive enhancement (no dependencies).

// Reveal-on-scroll for sections.
const reveal = () => {
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.08 }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
};

// Active nav link highlighting based on scroll position.
const activeNav = () => {
  const links = [...document.querySelectorAll("nav a.navlink[href^='#']")];
  const map = new Map(
    links
      .map((l) => {
        const id = l.getAttribute("href").slice(1);
        const sec = document.getElementById(id);
        return sec ? [sec, l] : null;
      })
      .filter(Boolean)
  );
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        const link = map.get(e.target);
        if (!link) return;
        if (e.isIntersecting) {
          links.forEach((l) => (l.style.color = ""));
          link.style.color = "var(--white)";
        }
      });
    },
    { rootMargin: "-45% 0px -50% 0px" }
  );
  map.forEach((_, sec) => io.observe(sec));
};

document.addEventListener("DOMContentLoaded", () => {
  if ("IntersectionObserver" in window) {
    reveal();
    activeNav();
  } else {
    document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in"));
  }
});
