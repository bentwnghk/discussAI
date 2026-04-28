export const AUDIO_TTL_DAYS = Number(process.env.AUDIO_TTL_DAYS) || 365;
export const AUDIO_TTL_MS = AUDIO_TTL_DAYS * 24 * 60 * 60 * 1000;
