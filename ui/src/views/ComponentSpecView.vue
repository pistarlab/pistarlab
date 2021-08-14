<template lang="html">
<div >
    <h2><i class="fas fa-stream"></i>Component Spec</h2>
    <div class="mt-4"></div>

    <b-container fluid>
        <b-row>
            <b-col>
                <div class="pt-2">
                    <div class="data_label">Component Spec</div>
                    <span>{{ componentSpec.ident }}</span>
                </div>
                <div class="pt-2">
                    <div class="data_label">Extension ID</div>
                    <span>{{ componentSpec.extensionId }}</span>
                </div>
                <div class="pt-2">
                    <div class="data_label">Version</div>
                    <span>{{ componentSpec.version }}</span>
                </div>
                <div class="pt-2">
                    <div class="data_label">Entry Point</div>
                    <span>{{ componentSpec.entryPoint }}</span>
                </div>
                <div class="pt-2">
                    <div class="data_label">Description</div>
                    <span>{{ componentSpec.description }}</span>
                </div>
            </b-col>
        </b-row>
        <div class="mt-4"></div>

        <b-row>
            <b-col>
                <div class="data_label">Default Config</div>
                <div v-if="componentSpec && componentSpec.config">
                    <pre>{{JSON.parse(componentSpec.config)}}</pre>
                </div>
            </b-col>
        </b-row>
        <div class="mt-4"></div>

    </b-container>
</div>
</template>

<script>
//USING https://github.com/chairuosen/vue2-ace-editor
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    timedelta,
    timepretty
} from "../funcs";
import gql from "graphql-tag";

const GET_AGENT_SPEC = gql `
  query GetComponentSpec($ident: String!) {
    componentSpec(ident: $ident) {
      id
      ident
      entryPoint
      description
      extensionId
      version
      config
    }
  }
`;

export default {
    name: "ViewComponent",
    components: {
    },
    apollo: {
        componentSpec: {
            query: GET_AGENT_SPEC,
            variables() {
                return {
                    ident: this.specId,
                };
            },
        },
    },
    data() {
        return {
            componentSpec: {},
            options: {},
            config: "",
            code: '',
            submitting: false,
        };
    },
    props: {
        specId: String
    },
    methods: {
//
    },
    watch: {
//
    },
    created() {

        //
    },
};
</script>

<style scoped>
.ace_editor {
    font-size: 16px;
}
</style>
