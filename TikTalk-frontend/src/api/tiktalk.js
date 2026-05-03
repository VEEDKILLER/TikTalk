const BASE = '/api'

async function handleResponse(res) {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try { msg = (await res.json()).detail || msg } catch {}
    throw new Error(msg)
  }
  return res.json()
}

export async function getCategories() {
  return handleResponse(await fetch(`${BASE}/session/categories`))
}

export async function generateImage({ prompt, category, character, action }) {
  return handleResponse(await fetch(`${BASE}/session/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, category, character, action }),
  }))
}

export async function getGroundTruth(imageId) {
  return handleResponse(await fetch(`${BASE}/session/vlm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId }),
  }))
}

export async function evaluate(audioBlob, imageId, { ageGroup, gradeLevel } = {}) {
  const form = new FormData()
  form.append('audio', audioBlob, 'recording.webm')
  form.append('image_id', imageId)
  if (ageGroup)   form.append('age_group', ageGroup)
  if (gradeLevel) form.append('grade_level', gradeLevel)
  return handleResponse(await fetch(`${BASE}/session/evaluate`, {
    method: 'POST',
    body: form,
  }))
}
