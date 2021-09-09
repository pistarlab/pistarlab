<template>
<div v-bind:class="mainClass">
    <div v-if="show" class="small">
        <b-navbar>
            <b-link v-if="focus !='outline'" @click="focus='outline'"><i class="fa fa-chevron-up iconlink"></i></b-link>

            <b-link v-if="!fullPage" class="ml-auto mr-2 iconlink" size="sm" @click="toggle()"><i class="fa fa-times"></i></b-link>

        </b-navbar>

        <div v-if="focus =='outline'">
            <div class="ml-3">
                <h4>Help Contents</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>

                        <p>
                        </p>
                        <ul class="outline">
                            <li>
                                <b-link @click="focus='gettingstarted'">Getting Started</b-link>
                            </li>
                            <li>
                                <b-link @click="focus='agents'">Agents</b-link>
                            </li>
                            <li>

                                <b-link @click="focus='envs'">Environments</b-link>
                            </li>
                            <li>
                                <b-link @click="focus='extensions'">Extensions</b-link>
                            </li>
                            <li>
                                <b-link @click="focus='workspace'">Workspace</b-link>
                            </li>
                            <li>
                                <b-link @click="focus='sessions'">Sessions</b-link>
                            </li>
                            <li>
                                <b-link @click="focus='tasks'">Tasks</b-link>
                            </li>
                        </ul>
                        <a target="_blank" :href="appConfig.API_URL+'/docs/'">Open Documentation</a>

                    </b-col>
                </b-row>
            </b-container>
        </div>
        <div v-if="focus =='gettingstarted'">
            <div class="ml-3">
                <h4>Getting Started</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>

                        <b-alert show>
                            Note: Still in early development. Some functionality is limited or difficult to use.
                        </b-alert>

                        <h4>What can I do with piSTAR Lab?</h4>
                        <p>
                            piSTAR Lab is an experimentation platform designed to help you create and train your own Agents on a growing number of environments. Depending on your background and motivation, you can use piSTAR Lab in different ways.
                            <br />

                            <ul>
                                <li><b>- AI Enthusiasts</b> that have limited or no coding experience, can use piSTAR Lab to train Agents with a few clicks in the UI. Like teaching a child, you can train your agent on simple related tasks first to prepare it for more difficult tasks. Competitions are coming.</li>
                                <li><b>- AI Researchers and Engineers</b> can develop new or add their existing algorithms to piSTAR via "Extensions"</li>
                                <li><b>- Game Developers</b> intersted in AI can create Extensions" which make there game playable by piSTAR Agents.</li>
                            </ul>

                        </p>

                        <br />

                    </b-col>
                </b-row>
            </b-container>
        </div>
        <div v-if="focus =='community'">
            <div class="ml-3">
                <h4>Community Hub</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>

                        <p>
                            Published agents are shared with other piSTAR.ai users.

                        </p>

                        <br />

                    </b-col>
                </b-row>
            </b-container>
        </div>
        <div v-if="focus =='agents'">
            <div class="ml-3">
                <h4>Agents</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>

                        <br />

                        <p>
                            An Agent is a configured instance of an "Agent Spec" or learning algorithm. Agents gain experience by interacting with environments.
                        </p>

                        <br />

                        <b-link size="sm" @click="focus='agents_details'">more</b-link>
                    </b-col>
                </b-row>
            </b-container>
        </div>
        <div v-if="focus =='agents_details'">
            <div class="ml-3">
                <h4>Agent Details</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>

                        <br />

                        <h6>Checkpoints</h6>
                        <p>
                            Checkpoints are the saved state of an agent. Checkpoints are created periodically by the task.

                        </p>
                        <br />
                        <h6>Snapshots</h6>
                        <p>
                            Agents can be expored into "Snapshots" for archiving or pubishing to a shared repository. Snapshots contain a recent checkpoint, configuration, and other useful metadata.
                        </p>
                        <br />

                        <h6>Agent Stats</h6>
                        <p>
                            Statistics gathered while an agent is running usually during the learning step.
                        </p>
                        <br />
                        <b-link @click="focus='agentenv'">Agent /Environment Compatibility</b-link>
                        <div class="text-center">
                            <img src="ai-play_5.png" width=200 />
                        </div>
                    </b-col>
                </b-row>
            </b-container>
        </div>
        <div v-if="focus =='agentenv'">
            <div class="ml-3">
                <h4>Agent Tasks</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>

                        <h6>Agent-environment compatibility</h6>
                        <p>
                            Not all agents or agent specs are compatible with all environments. The observations an agent receives from an environment but be supported by that agent. Likewise, the the actions taken by the agent must be supported by the environment.
                        </p>

                    </b-col>
                </b-row>
            </b-container>
        </div>

        <div v-if="focus =='agent_interface'">
            <div class="ml-3">
                <h4>Agent Interface</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>
                        <p>
                            Interfaces define how an agent can interact with an environment. Agents that support multiple interfaces can interact with environments with different obvservation/action spaces. For example: an RL agent that can learn from prior gathered experiences as well as run on the environment directly.
                        </p>
                    </b-col>
                </b-row>
            </b-container>
        </div>

        <div v-else-if="focus =='envs'">
            <div class="ml-3">
                <h4>Environments</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>

                        <br />
                        <p>
                            An environment is a collection of one or more "Environment Specs". Agents are tasked with interacting with Environments Specs usually with goal defined by the environment.
                            The differences between Environment Specs within the same Environment can vary greatly.

                        </p>
                        <br />
                        <h6>New Environments</h6>
                        <p>
                            New environments can be installed with the <b-link to="/extension/home">Extension Manager</b-link>
                        </p>
                        <br />
                        <h6>Environment Specs</h6>
                        <p>
                            We currenlty support Single-Agent and Multi-Agent Reinforcement Learning environments.
                            <br />
                            <br />
                            <b>Data Environments</b> are coming soon.
                        </p>

                        <br />
                        <h6>Rewards</h6>
                        <p>
                            Rewards are used to provide guidence to the agent while interacting with an environment.
                        </p>

                    </b-col>
                </b-row>
            </b-container>
        </div>
        <div v-else-if="focus =='workspace'">
            <div class="ml-3">
                <h4>Workspace</h4>
            </div>
            <b-container>
                <b-row>
                    <b-col>
                        <p>
                            Your workspace is where you can create and develop your own custom extensions.
                            By default, your workspace directory is under you the pistarlab/workspace folder in your home directory.
                        </p>
                        <br />
                        <h6>Backup and Version Control</h6>
                        <p>
                            You are responsible for backing up extensions you create in your workspace.
                        </p>
                    </b-col>
                </b-row>
            </b-container>

        </div>
        <div v-else-if="focus =='tasks'">
            <div class="ml-3">
                <h4>Tasks</h4>
            </div>
            <b-container>
                <b-row>
                    <b-col>
                        <p>
                            Tasks are used to start sessions or other child tasks. All Sessions are managed by a task and some tasks are launched by parent tasks.
                        </p>
                        <p>
                            Tasks can be run multiple times which is very useful for debugging code.
                        </p>
                        <br />
                        <h6>Task Specs</h6>
                        <p>
                            Task types are called Task Specs. There are multiple builtin task specs as well as support for adding new task types via "extentions".

                        </p>
                        <p>
                            Built-in Task Specs:
                            <ul>
                                <li>
                                    Agent Tasks: The main type of task used when assigning agents to interact with environments. Supports multi-agent environments when only one agent instance is used.
                                </li>
                                <li>
                                    Multi-Agent Tasks: This task spec used when assigning different agents to interact with the same environment. When a multi-agent task is created, an Agent task is created for each agent. Agent tasks then communicate with the Multi-Agent task over a network connection.
                                </li>
                            </ul>
                        </p>

                    </b-col>
                </b-row>
            </b-container>

        </div>
        <div v-else-if="focus =='sessions'">
            <div class="ml-3">
                <h4>Sessions</h4>
            </div>
            <b-container>
                <b-row>
                    <b-col>
                        <p>
                            Sessions represent an interaction between an agent and an environment. Sessions are used to collect and manage the interaction data. A session is started and managed by a "task". When a task terminates, all sessions are aborted.

                        </p>
                        <p>

                        </p>
                        <br />
                        <h6>Session Types</h6>
                        <p>
                            Different types of sessions are used for different types of environment interactions. For example, single agent sessions collect data about a solo agent's interactions while multi agent sessions capture the group interaction data.

                        </p>

                    </b-col>
                </b-row>
            </b-container>

        </div>

        <div v-else-if="focus =='extensions'">
            <div class="ml-3">
                <h4>Extensions</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>
                        Use extensions to add new Agents, Environments, and other functionality.

                        <br />
                        <br />
                        <h6>Workspace</h6>
                        <p>
                            Your workspace is where you can create and develop your own custom extensions.
                            By default, your workspace directory is under you the pistarlab/workspace folder in your home directory.
                        </p>

                        <br />

                        <h6>Repositories</h6>
                        There are three predefined extension repositories:
                        <p>
                            <ul>
                                <li>
                                    "builtin": these are local extensions included with installation
                                </li>
                                <li>
                                    "workspace": extensions located in your pistarlab workspace. This is a local repo for extensions developed by the user (you).
                                </li>
                                <li>
                                    "main": extensions located in the the remote pistarlab-repo project on github. See https://github.com/pistarlab/pistarlab-repo/main/extensions
                                </li>
                            </ul>
                        </p>
                        To add additional sources, you can make changes to the pistarlab/data/extension_sources.json file.
                        <br />
                        Source types are defined as follows:
                        <br />

                        <ul>
                            <li>
                                "path": scans path for extensions
                            </li>
                            <li>
                                "workspace": scans workspace path for extensions.
                            </li>
                            <li>
                                "file": extensions are defined in a local repo.json file
                            </li>
                            <li>
                                "remote": extensions are defined in a remote repo.json file
                            </li>
                        </ul>

                    </b-col>
                </b-row>
            </b-container>
        </div>
    </div>
    <div v-else>
        <b-link size="sm" @click="toggle()"><i class="fa fa-question iconlink"></i></b-link>
    </div>
</div>
</template>

<script>
import axios from "axios";
import {
    appConfig
} from "../app.config";
import gql from "graphql-tag";
export default {
    props: {
        contentId: String,
        fullPage: {
            type: Boolean,
            default: false
        }
    },
    data() {
        return {
            mainClass: "help-content",
            show: true,
            focus: ""

        };
    },
    mounted() {
        //
    },
    methods: {
        //
        toggle() {

            this.show = !this.show
            if (this.show) {
                this.mainClass = "help-content"
            } else {
                this.mainClass = "help-content-collapsed"
            }

        }

    },
    computed: {
        //

    },
    created() {
        this.focus = this.contentId
        if (this.fullPage == true) {
            this.mainClass = "help-content-fullpage"
        } else {
            this.fullPage = false
        }

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
