---
layout: default
title: ACTING-MACHINES
description: Robotics research blog
projects:
  - title: "Can an Autoregressive VLM Control a Robot in Real Time? (Part 1)"
    url: "/ar_vla_real_time_p1.html"
    date: "2026-05-20"
    authors:
      - "Oleg Balakhnov"
      - "Sergei Skvortsov"
    summary: "A systems-focused look at reducing action-generation latency for autoregressive vision-language-action robot control."
    tags:
      - "VLA"
      - "real-time control"
      - "robotics"
---

<style>
  .project-list {
    display: grid;
    gap: 18px;
    margin-top: 20px;
  }

  .project-card {
    padding: 18px 20px;
    border: 1px solid #dbe3ef;
    border-radius: 8px;
    background: #ffffff;
  }

  .project-card h2 {
    margin-top: 0;
    margin-bottom: 8px;
    font-size: 1.3rem;
  }

  .project-meta {
    margin: 0 0 12px;
    color: #64748b;
    font-size: 0.95rem;
  }

  .project-summary {
    margin-bottom: 14px;
  }

  .project-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .project-tags li {
    padding: 3px 8px;
    border-radius: 999px;
    background: #eef2ff;
    color: #3730a3;
    font-size: 0.85rem;
  }
</style>

# Our projects

<div class="project-list">
{% for project in page.projects %}
  <article class="project-card">
    <h2><a href="{{ project.url | relative_url }}">{{ project.title }}</a></h2>
    <p class="project-meta">
      {{ project.date | date: "%B %-d, %Y" }} &middot; {{ project.authors | join: ", " }}
    </p>
    <p class="project-summary">{{ project.summary }}</p>
    {% if project.tags %}
    <ul class="project-tags" aria-label="Project tags">
      {% for tag in project.tags %}
      <li>{{ tag }}</li>
      {% endfor %}
    </ul>
    {% endif %}
  </article>
{% endfor %}
</div>

<!--
Template for another project:

projects:
  - title: "Project title"
    url: "/project-page.html"
    date: "YYYY-MM-DD"
    authors:
      - "Author Name"
    summary: "One sentence describing the project."
    tags:
      - "tag"
-->
