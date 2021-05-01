<template>
<div>

    <h1><i class="fa fa-home"></i> Home</h1>

    <b-container>
        <b-row>
            <b-col>

                <div>
                    <p>

                    </p>
                </div>

            </b-col>
        </b-row>

        <b-row class="text-center align-items-center">

            <b-col>
                <b-button id="newagenttaskbutton" variant="info" size="lg" to="task/new/agenttask" style="width:300px;height:100px">
                    <h3>
                        <div>
                            New Task
                        </div>
                        <i class="fa fa-plus"></i>
                    </h3>
                </b-button>
                                  <b-tooltip target="newagenttaskbutton" triggers="hover">
                    Launch a new single- or multi- agent task.
                </b-tooltip>
            </b-col>
        </b-row>
        <div style="height:100px">
        </div>
        <b-row class="text-center align-items-center">

            <b-col>
                <b-button id="agentlibbutton" variant="info" size="lg" to="agent/home" style="width:400px;height:150px">
                    <h1>
                        <div>
                            Agent Library
                        </div>
                        <i class="fa fa-robot"></i>
                    </h1>
                </b-button>
                    <b-tooltip target="agentlibbutton" triggers="hover">
                    View and create agent instances.
                </b-tooltip>
            </b-col>

            <b-col>
                <b-button id="envlibbutton" variant="info" size="lg" to="env/home" style="width:400px;height:150px">
                    <h1>
                        <div>
                            Environment Library
                        </div>
                        <i class="fa fa-gamepad"></i>
                    </h1>
                </b-button>
                    <b-tooltip target="envlibbutton" triggers="hover">
                    Explore currently installed Environments.
                </b-tooltip>
            </b-col>
        </b-row>
        <div style="height:100px">
        </div>
        <b-row class="text-center align-items-center">

            <b-col>
                <b-button id="pluginbutton" variant="secondary" size="lg" v-b-modal.plugin-manager style="width:400px;height:100px">
                    <h3>
                        <div> Plugins</div>
                        <i title="Plugin Manager" class="fa fa-cogs"></i>
                    </h3>
                </b-button>
                <b-tooltip target="pluginbutton" triggers="hover">
                    Manage plugins to access new types of Agents and Environments
                </b-tooltip>
            </b-col>

        </b-row>
    </b-container>

    <!-- <b-card title="Missions in Progress">
        <b-card-text>
            <ul>
                <li>Solve Cartpole: Create an Agent which can solve Cartpole-v1 in less than 50k timesteps.</li>
            </ul>

        </b-card-text>
    </b-card>

    <b-card title="My Achievments">
        <b-card-text>
            <div>No achievements yet</div>

        </b-card-text>
    </b-card> -->

    <!-- <div class="mt-4">
    </div> -->

</div>
</template>

<script>
import {
    appConfig
} from "../app.config";

export default {
    name: "Home",
    components: {

    },
    apollo: {
        //

    },
    data() {
        return {
            workspace: null,
            projectName: "default",
            packageName: "",
            message: ".",
            appConfig
        };
    },
    methods: {

    },

    computed: {
        //
    },
    created() {
//
    },
};
</script>

<style >
</style>
