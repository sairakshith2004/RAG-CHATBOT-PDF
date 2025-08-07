// api/chat.js
export default async function handler(req, res) {
const response = await callYourRAGModel(req.body.query);
res.json({ answer: response });
}