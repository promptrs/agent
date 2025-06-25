#[allow(warnings)]
mod bindings;

use bindings::exports::promptrs::agent::runner::Guest;
use bindings::promptrs::client::completion::{
	Message, Params, Request, Response as RawResponse, ToolCall, receive,
};
use bindings::promptrs::parser::response::{Delims, parse};
use bindings::promptrs::tools::caller::{System, ToolDelims, Tooling};
use serde::Deserialize;

struct Component;

impl Guest for Component {
	fn run(input: String, config: String) -> String {
		let tooling = Tooling::new(&config);
		let Ok(config) = serde_json::from_str::<Config>(&config) else {
			return "Failed".into();
		};

		let delims = &Delims {
			reasoning: config.delims.reasoning,
			tool_call: config.delims.tool_call.clone(),
		};
		let System {
			prompt,
			status_call,
		} = tooling.init(&ToolDelims {
			available_tools: config.delims.available_tools,
			tool_call: config.delims.tool_call,
		});

		let user = match tooling.prompt(&input) {
			Ok(output) => return output,
			Err(user) => user,
		};
		let mut request = Request {
			api_key: config.api_key,
			base_url: config.base_url,
			body: Params {
				model: config.model,
				temperature: config.temperature,
				top_p: config.top_p,
				messages: vec![Message::System(prompt), Message::User(user)],
				stream: true,
			},
		};

		loop {
			let Ok(RawResponse {
				mut text,
				mut tool_calls,
			}) = receive(&request)
			else {
				continue;
			};

			if tool_calls.is_empty() {
				let parsed = parse(&text, Some(&delims));
				text = parsed.content;
				tool_calls = parsed
					.tool_calls
					.into_iter()
					.map(|tc| ToolCall {
						name: tc.name,
						arguments: tc.arguments,
					})
					.collect();
			}

			let text = text.trim();
			if !text.is_empty() {
				request.body.messages.push(Message::Assistant(text.into()));
			}

			for tool_call in tool_calls {
				let resp = tooling.call(&tool_call.name, &tool_call.arguments);
				let tc = format!(
					r#"{{"name":"{}","arguments":{}}}"#,
					tool_call.name, tool_call.arguments
				);

				let messages = &mut request.body.messages;
				messages.push(Message::ToolCall((tc, resp.output)));
				if let Some(status) = resp.status {
					messages.push(Message::Status((status_call.clone(), status)));
				}
			}

			match tooling.prompt(text) {
				Ok(output) => return output,
				Err(user) => request.body.messages.push(Message::User(user)),
			};

			request.body.messages = prune(request.body.messages, 20000);
		}
	}
}

fn prune(messages: Vec<Message>, size: u64) -> Vec<Message> {
	let mut pruned = messages.iter().skip(1).rev().scan(0, |acc, msg| {
		if *acc > size as usize {
			return None;
		}
		*acc += match msg {
			Message::System(content) => *acc + content.len(),
			Message::User(content) => *acc + content.len(),
			Message::Assistant(content) => *acc + content.len(),
			Message::ToolCall((req, res)) => *acc + req.len() + res.len(),
			Message::Status((req, res)) => *acc + req.len() + res.len(),
		};
		Some(msg)
	});

	let mut messages = if let Some(pos) = pruned.position(is_status) {
		pruned
			.clone()
			.take(pos + 1)
			.chain(pruned.skip(pos + 1).filter(|msg| !is_status(msg)))
			.chain(messages.iter().take(1))
			.cloned()
			.collect::<Vec<_>>()
	} else {
		pruned.chain(messages.iter().take(1)).cloned().collect()
	};

	messages.reverse();
	messages
}

fn is_status(msg: &Message) -> bool {
	if let Message::Status(_) = msg {
		true
	} else {
		false
	}
}

#[derive(Deserialize)]
struct Config {
	base_url: String,
	api_key: Option<String>,
	model: String,
	temperature: Option<f64>,
	top_p: Option<f64>,
	delims: DelimConfig,
}

#[derive(Deserialize)]
struct DelimConfig {
	reasoning: Option<(String, String)>,
	available_tools: (String, String),
	tool_call: (String, String),
}

bindings::export!(Component with_types_in bindings);
