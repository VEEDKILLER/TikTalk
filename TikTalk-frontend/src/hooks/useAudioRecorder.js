import { useState, useRef, useCallback } from 'react'

export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState(null)
  const [duration, setDuration] = useState(0)
  const [error, setError] = useState(null)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const streamRef = useRef(null)

  const start = useCallback(async () => {
    setError(null)
    setAudioBlob(null)
    setDuration(0)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'

      const recorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType })
        setAudioBlob(blob)
        stream.getTracks().forEach((t) => t.stop())
      }

      recorder.start(250) // collect data every 250ms
      setIsRecording(true)

      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000)
    } catch (err) {
      setError('Microphone access denied. Please allow microphone access and try again.')
    }
  }, [])

  const stop = useCallback(() => {
    mediaRecorderRef.current?.stop()
    streamRef.current?.getTracks().forEach((t) => t.stop())
    clearInterval(timerRef.current)
    setIsRecording(false)
  }, [])

  const reset = useCallback(() => {
    stop()
    setAudioBlob(null)
    setDuration(0)
    setError(null)
  }, [stop])

  return { isRecording, audioBlob, duration, error, start, stop, reset }
}
