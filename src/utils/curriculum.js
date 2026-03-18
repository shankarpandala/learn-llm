/**
 * Curriculum navigation utilities for Learn-LLM.
 * Uses subjects/index.js as the authoritative data source.
 */
import { CURRICULUM } from '../subjects/index.js';

const curriculumData = { subjects: CURRICULUM };

/**
 * Get subject metadata by its ID string (e.g. '04-transformer-architecture').
 */
export function getSubject(id) {
  return CURRICULUM.find(s => s.id === id) || null;
}

/**
 * Get chapter metadata within a subject.
 */
export function getChapter(subjectId, chapterId) {
  const subject = getSubject(subjectId);
  if (!subject) return null;
  return (subject.chapters || []).find(c => c.id === chapterId) || null;
}

/**
 * Get section metadata within a chapter.
 */
export function getSection(subjectId, chapterId, sectionId) {
  const chapter = getChapter(subjectId, chapterId);
  if (!chapter) return null;
  return (chapter.sections || []).find(s => s.id === sectionId) || null;
}

/**
 * Build a flat ordered list of all sections across all chapters of a subject.
 */
function flatSections(subjectId) {
  const subject = getSubject(subjectId);
  if (!subject) return [];
  const flat = [];
  for (const chapter of subject.chapters || []) {
    for (const section of chapter.sections || []) {
      flat.push({ subjectId, chapterId: chapter.id, sectionId: section.id, ...section });
    }
  }
  return flat;
}

/**
 * Get the next section after the given one, or null if at the end.
 */
export function getNextSection(subjectId, chapterId, sectionId) {
  const flat = flatSections(subjectId);
  const idx = flat.findIndex(s => s.chapterId === chapterId && s.sectionId === sectionId);
  if (idx === -1 || idx >= flat.length - 1) return null;
  return flat[idx + 1];
}

/**
 * Get the previous section before the given one, or null if at the start.
 */
export function getPrevSection(subjectId, chapterId, sectionId) {
  const flat = flatSections(subjectId);
  const idx = flat.findIndex(s => s.chapterId === chapterId && s.sectionId === sectionId);
  if (idx <= 0) return null;
  return flat[idx - 1];
}

export default curriculumData;
