/**
 * Chargebee server-side client singleton.
 * Only import this in API routes / server components — never on the client.
 */
import Chargebee from "chargebee";

let instance: Chargebee | null = null;

export function getChargebee(): Chargebee {
  if (!instance) {
    instance = new Chargebee({
      site: process.env.CHARGEBEE_SITE!,
      apiKey: process.env.CHARGEBEE_API_KEY!,
    });
  }
  return instance;
}
