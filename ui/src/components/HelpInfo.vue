<template>
<div v-bind:class="mainClass">
    <div v-if="show" class="small">
        <b-navbar>
            <b-link v-if="focus !='outline'" @click="focus='outline'"><i class="fa fa-chevron-up iconlink"></i></b-link>

            <b-link v-if="!fullPage" class="ml-auto mr-2 iconlink" size="sm" @click="toggle()"><i class="fa fa-times"></i></b-link>

        </b-navbar>

        <div v-if="focus =='outline'">
            <div class="ml-3">
                <h4>Contents</h4>
            </div>

            <b-container>
                <b-row>
                    <b-col>

                        <p>
                        </p>
                        <ul class="outline">
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
                        </ul>

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

                        <h6>Agent-environment compatibility</h6>
                        <p>
                            Not all agents or agent specs are compatible with all environments. The observations an agent receives from an environment but be supported by that agent. Likewise, the the actions taken by the agent must be supported by the environment.
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
          <div class="text-center">
                         <img src="ai-play_5.png" width=200 />
                         </div>
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
                        <h6>Environment Specs</h6>
                        <p>
                            We currenlty support Single-Agent and Multi-Agent Reinforcement Learning environments.
                        </p>
                        <br />
                        <h6>Rewards</h6>
                        <p>
                            Rewards can be used provide guidence to the agent while interacting with an environment.
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
